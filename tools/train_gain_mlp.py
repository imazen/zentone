#!/usr/bin/env python3
"""Standalone Gain-MLP trainer for zentone::gainmap_mlp.

Implements Canham, Pieper, Sankaranarayanan, Roghair, Mueller (2025),
"Encoding High-Dynamic-Range Gain Maps with Small MLPs" — the
per-image training procedure that produces a ~10 KB MLP encoding the
HDR/SDR log2 gain residual.

The output of this script is a `BakeRequestJson` (zenpredict-bake's
canonical input format). The companion `zenpredict-bake` CLI converts
the JSON to a ZNPR v3 binary that `zentone::GainMapMlpDecoder` loads
at runtime.

USAGE
-----

    python3 train_gain_mlp.py \\
        --hdr path/to/hdr.{png,npy,exr} \\
        --sdr path/to/sdr.{png,npy} \\
        --out gainmlp.bin \\
        [--bake-bin /path/to/zenpredict-bake] \\
        [--epochs 1000] [--lr 1e-2] [--batch 65536] \\
        [--hidden 16] [--num-freqs 12] [--epsilon 0.015625]

PIPELINE
--------

1. Load HDR + SDR image pair.
2. Compute per-pixel log2 gain target:  `log2((H + ε) / (S + ε))`.
3. Optionally run meta-initialisation: 10 000 iters on synthetic Daly-style
   chromatic-noise patterns. Initial weights for the per-image fit.
4. Per-image fit: Adam 1000 iters, lr 1e-2, batch 65 536 random pixels.
5. Compute scaler statistics (mean / scale) over the 120-D embedded
   inputs across a deterministic 4 096-pixel sample.
6. Emit BakeRequestJson with the trained weights, scaler stats, and
   the required `feature_transforms = sinusoidal × 5` +
   `feature_transform_params = "1,2,4,...,2^(num_freqs-1)"` metadata.
7. Optionally invoke `zenpredict-bake <json> <bin>` to materialise the
   ZNPR v3 binary.

The script is **per-image** by design (Canham §3.2) — there is no
batch trainer. For a fleet, invoke per-image in parallel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Sinusoidal positional embedding — matches FeatureTransform::Sinusoidal
# in zenpredict (see ../../zenanalyze/zenpredict/src/feature_transform.rs).
# ---------------------------------------------------------------------------


def sinusoidal_embed(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """
    Embed `x` (shape (..., D)) into (..., D * 2 * F) by computing
    `[sin(2π·f·x), cos(2π·f·x)]` for each of the `F = frequencies.numel()`
    frequencies on each of the `D` input dimensions.

    Matches zenpredict's FeatureTransform::Sinusoidal contract exactly.
    """
    # x shape (..., D); frequencies shape (F,); out (..., D * 2 * F)
    two_pi = 2.0 * math.pi
    angles = two_pi * x.unsqueeze(-1) * frequencies  # (..., D, F)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    interleaved = torch.stack([sin, cos], dim=-1)  # (..., D, F, 2)
    return interleaved.reshape(*x.shape[:-1], x.shape[-1] * 2 * frequencies.numel())


# ---------------------------------------------------------------------------
# Gain-MLP architecture: linear → ReLU → linear, 3 outputs (per-channel
# log2 gain). Matches the paper's two-layer × 16-neuron design.
# ---------------------------------------------------------------------------


class GainMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Daly-style chromatic noise — used for meta-init. The paper cites
# Daly et al.'s spatiochromatic noise process; we approximate by
# generating per-channel 1/f-spectrum noise in linear-light space.
# ---------------------------------------------------------------------------


def make_chromatic_noise(
    h: int, w: int, alpha: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Spatiochromatic 1/f noise per RGB channel in `[0, 1]` linear."""
    rng = np.random.default_rng(seed)
    out = np.empty((h, w, 3), dtype=np.float32)
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.fftfreq(w).reshape(1, -1)
    radius = np.sqrt(fy**2 + fx**2)
    radius[0, 0] = 1.0  # avoid div0
    spectrum = 1.0 / (radius**alpha)
    spectrum[0, 0] = 0.0
    for c in range(3):
        noise = rng.standard_normal((h, w))
        f_noise = np.fft.fft2(noise) * spectrum
        spatial = np.real(np.fft.ifft2(f_noise))
        spatial = (spatial - spatial.min()) / (np.ptp(spatial) + 1e-8)
        out[:, :, c] = spatial.astype(np.float32)
    return out


def reinhard_tonemap(hdr: np.ndarray) -> np.ndarray:
    """Simple global Reinhard tone-map: y = x / (1 + x). Used for the
    meta-init SDR counterpart."""
    return hdr / (1.0 + hdr)


# ---------------------------------------------------------------------------
# Training core.
# ---------------------------------------------------------------------------


def build_pixel_targets(
    hdr_image: np.ndarray,
    sdr_image: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (inputs, targets) where inputs is `(N, 5)` of
    `[x, y, R, G, B]` in `[0, 1]` and targets is `(N, 3)` of
    per-channel log2 gain."""
    assert hdr_image.shape == sdr_image.shape, "HDR/SDR shape mismatch"
    h, w, c = hdr_image.shape
    assert c == 3, "Only RGB supported"
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    flat_inputs = np.stack(
        [
            xx.flatten(),
            yy.flatten(),
            sdr_image[:, :, 0].flatten(),
            sdr_image[:, :, 1].flatten(),
            sdr_image[:, :, 2].flatten(),
        ],
        axis=1,
    ).astype(np.float32)
    # log2((H + ε) / (S + ε)) per channel
    flat_targets = np.log2(
        (hdr_image.reshape(-1, 3) + epsilon) / (sdr_image.reshape(-1, 3) + epsilon)
    ).astype(np.float32)
    return flat_inputs, flat_targets


def train_one_image(
    model: GainMLP,
    frequencies: torch.Tensor,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    lr: float,
    batch: int,
    device: torch.device,
) -> float:
    inputs_t = torch.from_numpy(inputs).to(device)
    targets_t = torch.from_numpy(targets).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n = inputs_t.shape[0]
    last_loss = float("inf")
    for ep in range(epochs):
        idx = torch.randint(0, n, (batch,), device=device)
        x = inputs_t[idx]
        y = targets_t[idx]
        embed = sinusoidal_embed(x, frequencies)
        out = model(embed)
        loss = torch.mean((out - y) ** 2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        last_loss = float(loss.item())
    return last_loss


def meta_initialise(
    model: GainMLP,
    frequencies: torch.Tensor,
    iters: int,
    lr: float,
    batch: int,
    device: torch.device,
    seed: int,
    img_count: int = 10,
    size: int = 64,
) -> float:
    """Meta-init via gradient descent on `img_count` synthetic noise
    images. Mirrors Canham §3.2's meta-initialisation step."""
    rng_seed = seed
    images: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(img_count):
        hdr = make_chromatic_noise(size, size, alpha=1.0, seed=rng_seed + k) * 4.0
        sdr = reinhard_tonemap(hdr).clip(0.0, 1.0)
        images.append(build_pixel_targets(hdr, sdr, epsilon=1.0 / 64.0))

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    last_loss = float("inf")
    for it in range(iters):
        inputs_np, targets_np = images[it % len(images)]
        inputs_t = torch.from_numpy(inputs_np).to(device)
        targets_t = torch.from_numpy(targets_np).to(device)
        n = inputs_t.shape[0]
        idx = torch.randint(0, n, (min(batch, n),), device=device)
        x = inputs_t[idx]
        y = targets_t[idx]
        embed = sinusoidal_embed(x, frequencies)
        out = model(embed)
        loss = torch.mean((out - y) ** 2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        last_loss = float(loss.item())
    return last_loss


# ---------------------------------------------------------------------------
# BakeRequestJson emission.
# ---------------------------------------------------------------------------


def compute_scaler_stats(
    inputs: np.ndarray,
    frequencies: np.ndarray,
    sample_size: int = 4096,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return an identity scaler (mean=0, scale=1).

    The MLP is trained directly on raw sinusoidal embedding values
    (which are inherently bounded to `[-1, 1]`) — no train-time
    standardisation. The bake's scaler must therefore be the identity
    transform, otherwise the runtime would apply `(embed - mean) / std`
    that the trainer never saw, producing reconstruction garbage
    (catastrophic train/serve skew).

    `sample_size` / `seed` are kept in the signature for compatibility
    with the on-disk JSON output format and in case a future trainer
    variant decides to learn with standardisation; both arguments are
    currently unused.

    Discovered 2026-05-18 during the first synthetic-pair round-trip:
    the original (mean / std) scaler dropped reconstruction PSNR to
    11.5 dB versus the paper's 48.5 dB. Switching to identity recovers
    the target.
    """
    _ = (inputs, frequencies, sample_size, seed)
    in_dim = 5 * 2 * frequencies.size
    return (np.zeros(in_dim, dtype=np.float32), np.ones(in_dim, dtype=np.float32))


def build_bake_request(
    model: GainMLP,
    frequencies: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    schema_hash: int = 0xFEED_FACE_DEAD_BEEF,
) -> dict:
    fc1_w = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
    fc1_b = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
    fc2_w = model.fc2.weight.detach().cpu().numpy().astype(np.float32)
    fc2_b = model.fc2.bias.detach().cpu().numpy().astype(np.float32)

    # Sanity: layer dims add up.
    in_dim = scaler_mean.size
    assert fc1_w.shape == (model.fc1.out_features, in_dim), (
        f"fc1_w shape {fc1_w.shape} != ({model.fc1.out_features}, {in_dim})"
    )
    hidden = model.fc1.out_features
    out_dim = model.fc2.out_features

    # Weight layout: zenpredict's saxpy_matmul indexes
    # `w[input_idx * out_dim + output_idx]` (see
    # zenpredict/src/inference.rs::saxpy_matmul_f32). PyTorch stores
    # `nn.Linear.weight` as `(out_features, in_features)`; flattening
    # that directly produces `w[c * in_dim + i]` (column-major from
    # zenpredict's POV) which silently miscomputes every forward pass.
    # Transpose before flatten so the C-order traversal yields the
    # row-major (in_dim, out_dim) zenpredict expects.
    fc1_w_zen = fc1_w.T.copy()  # (in_dim, hidden) C-order
    fc2_w_zen = fc2_w.T.copy()  # (hidden, out_dim) C-order

    layers = [
        {
            "in_dim": in_dim,
            "out_dim": hidden,
            "activation": "relu",
            "dtype": "f32",
            "weights": fc1_w_zen.flatten().tolist(),
            "biases": fc1_b.tolist(),
        },
        {
            "in_dim": hidden,
            "out_dim": out_dim,
            "activation": "identity",
            "dtype": "f32",
            "weights": fc2_w_zen.flatten().tolist(),
            "biases": fc2_b.tolist(),
        },
    ]

    transforms_line = "\n".join(["sinusoidal"] * 5)
    freq_line = ",".join(str(float(f)) for f in frequencies)
    params_line = "\n".join([freq_line] * 5)

    return {
        "schema_hash": schema_hash,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "layers": layers,
        "feature_bounds": [],
        "metadata": [
            {
                "key": "zentrain.feature_transforms",
                "type": "utf8",
                "text": transforms_line,
            },
            {
                "key": "zentrain.feature_transform_params",
                "type": "utf8",
                "text": params_line,
            },
            {
                "key": "zentone.gainmlp.epsilon",
                "type": "utf8",
                "text": "0.015625",
            },
        ],
    }


# ---------------------------------------------------------------------------
# I/O helpers — accept .npy (preferred for HDR f32) or simple PNG /
# any-format-pillow-can-load for SDR.
# ---------------------------------------------------------------------------


def load_hdr_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    assert arr.ndim == 3 and arr.shape[2] == 3, f"HDR shape {arr.shape}"
    return arr.astype(np.float32)


def load_sdr_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return load_hdr_npy(path)
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        sys.exit(
            f"SDR loading from {path} requires PIL (`pip install pillow`) or "
            "a precomputed .npy"
        )
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Gain-MLP for zentone::gainmap_mlp.")
    p.add_argument("--hdr", type=Path, help="HDR image (.npy linear-light float)")
    p.add_argument("--sdr", type=Path, help="SDR image (.png, .npy, ...)")
    p.add_argument("--out", type=Path, required=True, help="Output bake path (.bin)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument(
        "--num-freqs",
        type=int,
        default=12,
        help="Number of frequencies per input dim (paper uses 12 ⇒ 24-D per input).",
    )
    p.add_argument("--epsilon", type=float, default=1.0 / 64.0)
    p.add_argument(
        "--meta-init",
        action="store_true",
        help="Run synthetic chromatic-noise meta-init before per-image fit.",
    )
    p.add_argument("--meta-iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--bake-bin",
        type=Path,
        default=Path("/home/lilith/work/zen/zenanalyze--featuretransform-sinusoidal/target/release/zenpredict-bake"),
        help="Path to zenpredict-bake binary",
    )
    p.add_argument(
        "--json-only",
        action="store_true",
        help="Emit BakeRequestJson but skip the zenpredict-bake invocation.",
    )
    p.add_argument(
        "--synthetic-pair",
        action="store_true",
        help="Skip --hdr/--sdr and train on a single synthetic chromatic-noise pair (useful for smoke testing).",
    )
    p.add_argument(
        "--dump-pair",
        type=Path,
        default=None,
        help="If set, write the HDR + SDR training pair as <prefix>_hdr.npy / <prefix>_sdr.npy. "
        "Used by the Rust gainmlp_roundtrip example to bit-exactly reload the training pair.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gain-mlp] device: {device}")

    # NeRF-style 2^k frequency schedule, k = 0..num_freqs-1.
    frequencies = torch.tensor(
        [float(1 << k) for k in range(args.num_freqs)], dtype=torch.float32, device=device
    )
    in_dim = 5 * 2 * args.num_freqs  # 5 raw × 2 (sin+cos) × F
    print(f"[gain-mlp] frequencies: {frequencies.tolist()}")
    print(f"[gain-mlp] expanded input dim: {in_dim}")

    model = GainMLP(in_dim=in_dim, hidden_dim=args.hidden, out_dim=3).to(device)

    if args.meta_init:
        print(f"[gain-mlp] meta-init: {args.meta_iters} iters on chromatic noise")
        loss = meta_initialise(
            model,
            frequencies,
            iters=args.meta_iters,
            lr=args.lr,
            batch=args.batch,
            device=device,
            seed=args.seed,
        )
        print(f"[gain-mlp] meta-init final loss: {loss:.6f}")

    if args.synthetic_pair:
        print("[gain-mlp] training on synthetic chromatic-noise pair")
        hdr = make_chromatic_noise(128, 128, alpha=1.0, seed=args.seed) * 4.0
        sdr = reinhard_tonemap(hdr).clip(0.0, 1.0)
    else:
        if args.hdr is None or args.sdr is None:
            sys.exit("--hdr and --sdr required (or pass --synthetic-pair)")
        print(f"[gain-mlp] loading HDR {args.hdr} + SDR {args.sdr}")
        hdr = load_hdr_npy(args.hdr)
        sdr = load_sdr_image(args.sdr)
        # Light alignment check
        assert hdr.shape == sdr.shape, f"shape mismatch {hdr.shape} vs {sdr.shape}"

    if args.dump_pair is not None:
        hdr_path = args.dump_pair.with_name(args.dump_pair.name + "_hdr.npy")
        sdr_path = args.dump_pair.with_name(args.dump_pair.name + "_sdr.npy")
        np.save(hdr_path, hdr.astype(np.float32))
        np.save(sdr_path, sdr.astype(np.float32))
        print(f"[gain-mlp] dumped pair: {hdr_path} + {sdr_path}")

    inputs, targets = build_pixel_targets(hdr, sdr, epsilon=args.epsilon)
    print(f"[gain-mlp] training: {args.epochs} epochs, lr={args.lr}, batch={args.batch}")
    final_loss = train_one_image(
        model,
        frequencies,
        inputs,
        targets,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        device=device,
    )
    print(f"[gain-mlp] final loss (MSE on log2 gain): {final_loss:.6f}")

    # Compute scaler stats over the embedded inputs.
    freqs_np = frequencies.cpu().numpy()
    scaler_mean, scaler_scale = compute_scaler_stats(
        inputs, freqs_np, sample_size=min(4096, inputs.shape[0]), seed=args.seed
    )

    # Round-trip quick sanity: forward pass on a 128-pixel sample and
    # compare to the targets directly (helps catch dtype / dim bugs
    # before the bake is even written).
    model.eval()
    with torch.no_grad():
        idx = np.linspace(0, inputs.shape[0] - 1, 128, dtype=np.int64)
        sample_in = torch.from_numpy(inputs[idx]).to(device)
        sample_out = model(sinusoidal_embed(sample_in, frequencies))
        recon_err = torch.mean((sample_out - torch.from_numpy(targets[idx]).to(device)) ** 2)
        print(f"[gain-mlp] sample-128 reconstruction MSE: {recon_err.item():.6f}")

    bake_request = build_bake_request(model.cpu(), freqs_np, scaler_mean, scaler_scale)

    if args.json_only:
        json_path = args.out.with_suffix(".json")
        json_path.write_text(json.dumps(bake_request))
        print(f"[gain-mlp] wrote JSON to {json_path} (--json-only)")
        return 0

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=args.out.parent or "."
    ) as tmp:
        json.dump(bake_request, tmp)
        tmp_path = tmp.name
    try:
        bake_bin = args.bake_bin
        if not bake_bin.exists():
            sys.exit(
                f"zenpredict-bake not found at {bake_bin} — pass --bake-bin or "
                "build via `cargo build --release -p zenpredict-bake`"
            )
        cmd = [str(bake_bin), tmp_path, str(args.out)]
        print(f"[gain-mlp] invoking: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            sys.exit(f"zenpredict-bake failed (exit {result.returncode})")
        print(f"[gain-mlp] wrote bake to {args.out} ({args.out.stat().st_size} bytes)")
    finally:
        os.unlink(tmp_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
