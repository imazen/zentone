// Research tool — type ergonomics matter less than reproducibility.
#![allow(clippy::type_complexity)]

//! Offline coefficient generator for the BT.2446-A SIMD kernel's
//! `pow_midp_unchecked(1/2.4)` input transfer. The investigation in
//! `benchmarks/bt2446a_throughput_2026-06-20.md` ("polyfit replacement for
//! `x^(1/2.4)`" section) used this tool to verify that no polyfit-derived
//! polynomial replacement is throughput-positive vs the baseline.
//!
//! Run with `cargo run --example fit_pow_inv_24 --features __polyfit-tools
//! --release`. The output is a block of `const [f32; N]` arrays meant to
//! be pasted into `src/simd/curves.rs` if a future polyfit/magetypes update
//! changes the precision/throughput balance.
//!
//! What the four strategies probe:
//!   - **A**: plain monomial of `x` over `[0, hi]`. Confirms the prior
//!     pass's "10%+ error" claim for direct polynomial fits.
//!   - **B**: plain monomial of `√x` (sqrt substitution flattens the
//!     tangent at z=0). Best single-piece result is 5e-3 at deg 9 —
//!     can't push lower because `ChebyshevFit::as_monomial()` loses
//!     precision past ~deg 12 on wide domains.
//!   - **C**: rational `P(√x)/Q(√x)` with `P(0)=0` constraint. Worse than
//!     plain poly because the SK iteration drifts at the z=0 boundary
//!     even with the constraint (constraint is a soft weight, not a
//!     hard projection).
//!   - **D**: two-piece monomial split `z ∈ [0, ε] ∪ [ε, √hi]`. Reaches
//!     3.4e-4 max abs on the polynomial alone at `(eps=0.07, near deg 7,
//!     bulk deg 10)`. Kernel-amplified output error is 5.2e-4 — just
//!     fails the parity test by ~4%.
//!   - **E**: three-piece monomial split. Reaches 2.6e-4 polynomial / 2.2e-4
//!     kernel error — passes parity but regresses throughput from 258 →
//!     203 Mpix/s because of FMA pipe + register pressure.
//!
//! See the benchmark log for the full findings.

use polyfit::rational::{
    Constraint, ConstraintMode, ErrorWeighting, F32ScoreMetric, F32SearchConfig, F32SearchResult,
    RationalFit, RationalFitOptions,
};

// ---------------------------------------------------------------------------
// f32 measurement helpers (mirror polyfit's optimize_f32 evaluation path).
// ---------------------------------------------------------------------------

fn eval_f32_horner(x: f32, p: &[f32], q: &[f32]) -> f32 {
    let mut yp = *p.last().unwrap();
    for &c in p[..p.len() - 1].iter().rev() {
        yp = yp.mul_add(x, c);
    }
    let mut yq = *q.last().unwrap();
    for &c in q[..q.len() - 1].iter().rev() {
        yq = yq.mul_add(x, c);
    }
    yp / yq
}

fn eval_poly_f32_horner(x: f32, c: &[f32]) -> f32 {
    let mut y = *c.last().unwrap();
    for &ci in c[..c.len() - 1].iter().rev() {
        y = y.mul_add(x, ci);
    }
    y
}

fn ulp_dist(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let map = |v: f32| -> i64 {
        let bits = v.to_bits() as i32;
        i64::from(if bits < 0 { i32::MIN - bits } else { bits })
    };
    (map(a) - map(b)).unsigned_abs() as u32
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    max_ulp: u32,
    avg_ulp: f64,
    max_abs: f64,
    avg_abs: f64,
    max_rel: f64,
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_ulp={:>6} avg_ulp={:>9.2} max_abs={:.3e} avg_abs={:.3e} max_rel={:.3e}",
            self.max_ulp, self.avg_ulp, self.max_abs, self.avg_abs, self.max_rel,
        )
    }
}

fn measure_dense(
    eval: impl Fn(f32) -> f32,
    reference: impl Fn(f64) -> f64,
    lo: f32,
    hi: f32,
    n: usize,
) -> Metrics {
    let mut m = Metrics::default();
    let mut sum_ulp = 0u64;
    let mut sum_abs = 0.0;
    for i in 0..=n {
        let x = lo as f64 + (hi as f64 - lo as f64) * i as f64 / n as f64;
        let got = eval(x as f32);
        let expected = reference(x) as f32;
        let ulp = ulp_dist(got, expected);
        let abs_err = (f64::from(got) - f64::from(expected)).abs();
        let rel_err = if expected.abs() > 1e-6 {
            abs_err / f64::from(expected.abs())
        } else {
            0.0
        };
        m.max_ulp = m.max_ulp.max(ulp);
        m.max_abs = m.max_abs.max(abs_err);
        m.max_rel = m.max_rel.max(rel_err);
        sum_ulp += u64::from(ulp);
        sum_abs += abs_err;
    }
    m.avg_ulp = sum_ulp as f64 / (n + 1) as f64;
    m.avg_abs = sum_abs / (n + 1) as f64;
    m
}

fn print_const(name: &str, doc: &str, coeffs: &[f32]) {
    println!("/// {doc}");
    println!("const {name}: [f32; {}] = [", coeffs.len());
    for (i, &c) in coeffs.iter().enumerate() {
        let comma = if i + 1 < coeffs.len() { "," } else { "" };
        println!("    {c:>20.10e}{comma}");
    }
    println!("];");
}

// ---------------------------------------------------------------------------
// Reference: f(x) = x^(1/2.4)
// ---------------------------------------------------------------------------

const INV_24: f64 = 1.0 / 2.4;

fn ref_pow(x: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { x.powf(INV_24) }
}

fn ref_pow_sqrt(z: f64) -> f64 {
    if z <= 0.0 { 0.0 } else { z.powf(2.0 / 2.4) }
}

// ---------------------------------------------------------------------------
// Domain. We fit out to x=8 (z=2.83) so spec-out-of-range pixels stay accurate.
// ---------------------------------------------------------------------------

// HDR test inputs in the property + random rows go up to `~10` (max value
// per channel in the random_strip generator). We fit to 11 — just above
// the test ceiling — because going wider degrades the polynomial fit
// quality without adding any useful production range.
const DOMAIN_HI_X: f64 = 11.0;
const SWEEP_LO: f32 = 0.0;
const SWEEP_HI: f32 = 11.0;

// ---------------------------------------------------------------------------
// Plain monomial poly via Chebyshev fit + monomial conversion. Numerically
// stable up to about deg 12 on our domain; higher degrees see the
// Chebyshev→monomial conversion magnify f64 round-off because monomial
// coefficients alternate large opposing signs.
// ---------------------------------------------------------------------------

fn fit_plain_poly_for(
    f: impl Fn(f64) -> f64,
    domain_lo: f64,
    domain_hi: f64,
    degree: usize,
) -> Vec<f32> {
    use polyfit::ChebyshevFit;

    let n = 16_000;
    let pi = std::f64::consts::PI;
    let data: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let theta = pi * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
            let x = (domain_lo + domain_hi) / 2.0 + (domain_hi - domain_lo) / 2.0 * theta.cos();
            (x, f(x))
        })
        .collect();

    let fit = ChebyshevFit::new(data, degree).expect("Chebyshev fit");
    let poly = fit.as_monomial().expect("monomial conversion");
    poly.coefficients().iter().map(|&c| c as f32).collect()
}

// ---------------------------------------------------------------------------
// Rational fit on √x with `P(0)/Q(0) = 0` constraint (low-z behaviour).
// ---------------------------------------------------------------------------

fn fit_rational_sqrt_constrained(p_deg: usize, q_deg: usize, restarts: usize) -> F32SearchResult {
    let z_hi = DOMAIN_HI_X.sqrt();
    let options = RationalFitOptions {
        weighting: ErrorWeighting::Relative,
        n_samples: 40_000,
        restarts,
        // Pin (0, 0) hard so the boundary doesn't drift. The kernel pre-clamps
        // inputs to `pos_eps` (positive normal), so the fit at exactly 0 is
        // never evaluated on the actual hot path, but the f32 search wants a
        // well-behaved fit there to avoid huge boundary_ulp numbers that
        // mislead the local search.
        constraints: vec![Constraint::with_weight(0.0, 0.0, 1_000_000.0)],
        constraint_mode: ConstraintMode::Both,
        ..RationalFitOptions::default()
    };
    let fit = RationalFit::from_function(ref_pow_sqrt, 0.0..=z_hi, p_deg, q_deg, options)
        .expect("rational fit");
    let config = F32SearchConfig {
        metric: F32ScoreMetric::AbsoluteError,
        samples: Some(300_000),
        ..F32SearchConfig::default()
    };
    fit.as_polynomial()
        .optimize_f32(ref_pow, SWEEP_LO, SWEEP_HI, |x| x.sqrt(), config)
}

// ---------------------------------------------------------------------------
// Piecewise: split at z = ε. Below ε use a low-degree fit; above ε use the
// well-behaved monomial-on-√x. The split point ε is chosen so the piecewise
// continuity gap stays inside the f32 ULP grid.
// ---------------------------------------------------------------------------

fn piecewise_eval(z: f32, eps: f32, near: &[f32], far: &[f32]) -> f32 {
    if z < eps {
        eval_poly_f32_horner(z, near)
    } else {
        eval_poly_f32_horner(z, far)
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("===========================================================");
    println!("  Fitting f(x) = x^(1/2.4) over x ∈ [0, {}]", DOMAIN_HI_X);
    println!(
        "  Sweep [{}, {}], measurement at 1M dense samples",
        SWEEP_LO, SWEEP_HI
    );
    println!("===========================================================");

    let z_hi = DOMAIN_HI_X.sqrt();

    // -------------------------------------------------------------------
    // Pass A: plain monomial on √x at every workable degree.
    // -------------------------------------------------------------------
    println!("\n--- A: monomial poly on √x, z ∈ [0, {z_hi:.4}] ---");
    let mut best_a: Option<(usize, Vec<f32>, Metrics)> = None;
    for &deg in &[5usize, 7, 9, 10, 11, 12, 13, 14] {
        let coeffs = fit_plain_poly_for(ref_pow_sqrt, 0.0, z_hi, deg);
        let m = measure_dense(
            |x| eval_poly_f32_horner(x.sqrt(), &coeffs),
            ref_pow,
            SWEEP_LO,
            SWEEP_HI,
            200_000,
        );
        println!("deg {:>2}: {m}", deg);
        if best_a
            .as_ref()
            .is_none_or(|(_, _, b)| m.max_abs < b.max_abs)
        {
            best_a = Some((deg, coeffs, m));
        }
    }

    // -------------------------------------------------------------------
    // Pass B: rational on √x with constrained P(0)/Q(0) = 0.
    // -------------------------------------------------------------------
    println!("\n--- B: rational P(√x)/Q(√x), P(0)=0 constraint ---");
    let mut best_b: Option<F32SearchResult> = None;
    for &(p_deg, q_deg) in &[
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (5, 4),
        (6, 5),
        (7, 6),
    ] {
        let result = fit_rational_sqrt_constrained(p_deg, q_deg, 8);
        let m = measure_dense(
            |x| eval_f32_horner(x.sqrt(), &result.numerator, &result.denominator),
            ref_pow,
            SWEEP_LO,
            SWEEP_HI,
            200_000,
        );
        println!("  ({p_deg}/{q_deg}): {m}");
        if best_b.as_ref().is_none_or(|b| {
            let bm = measure_dense(
                |x| eval_f32_horner(x.sqrt(), &b.numerator, &b.denominator),
                ref_pow,
                SWEEP_LO,
                SWEEP_HI,
                50_000,
            );
            m.max_abs < bm.max_abs
        }) {
            best_b = Some(result);
        }
    }

    // -------------------------------------------------------------------
    // Pass C: piecewise — small linear (degree-1) near 0 + the best-A
    // monomial on the bulk. Pick split ε where the linear segment's error
    // crosses the bulk's local error.
    // -------------------------------------------------------------------
    println!("\n--- C: piecewise (near-0 low-deg + bulk monomial on √x) ---");
    if let Some((_, ref far, _)) = best_a {
        for &eps in &[0.01_f32, 0.02, 0.05, 0.1] {
            // Near branch: fit deg-2 / deg-3 to z ∈ [0, eps].
            for &near_deg in &[2usize, 3, 4, 5] {
                let near_coeffs = fit_plain_poly_for(ref_pow_sqrt, 0.0, eps as f64, near_deg);
                let m = measure_dense(
                    |x| piecewise_eval(x.sqrt(), eps, &near_coeffs, far),
                    ref_pow,
                    SWEEP_LO,
                    SWEEP_HI,
                    200_000,
                );
                println!("  eps={eps:>5.3}, near_deg={near_deg}: {m}");
            }
        }
    }

    // -------------------------------------------------------------------
    // Pass D: monomial on √x, eps-gated bulk fit. Sweep eps to find the
    // best (eps, deg) pair — re-fitting the bulk on `[eps, z_hi]` only
    // sidesteps the singular tangent at z=0.
    // -------------------------------------------------------------------
    println!("\n--- D: monomial on √x on bulk only, (eps, deg) sweep ---");
    let mut best_d: Option<(usize, f32, Vec<f32>, Vec<f32>, Metrics)> = None;
    // Hand-picked winner config — see comments in this loop body. The 2-piece
    // gives us enough headroom for the parity test's error-amplification
    // chain (Cb/Cr cancellation downstream of pow_inv24).
    for &eps in &[0.07_f32, 0.05_f32] {
        let near_coeffs = fit_plain_poly_for(ref_pow_sqrt, 0.0, eps as f64, 7);
        for &deg in &[10usize, 11] {
            let bulk_coeffs = fit_plain_poly_for(ref_pow_sqrt, eps as f64, z_hi, deg);
            let m = measure_dense(
                |x| piecewise_eval(x.sqrt(), eps, &near_coeffs, &bulk_coeffs),
                ref_pow,
                SWEEP_LO,
                SWEEP_HI,
                500_000,
            );
            println!("  eps={eps:>5.3} deg {deg:>2}: {m}");
            if best_d
                .as_ref()
                .is_none_or(|(_, _, _, _, b)| m.max_abs < b.max_abs)
            {
                best_d = Some((deg, eps, near_coeffs.clone(), bulk_coeffs, m));
            }
        }
    }

    // -------------------------------------------------------------------
    // Pass E: three-piece split on √x — `[0, eps1] ∪ [eps1, eps2] ∪
    // [eps2, z_hi]`. The middle segment absorbs the curvature transition
    // that's still hard for a single 9-12 degree polynomial.
    // -------------------------------------------------------------------
    println!("\n--- E: three-piece split on √x ---");
    let mut best_e: Option<(f32, f32, Vec<f32>, Vec<f32>, Vec<f32>, Metrics)> = None;
    for &(eps1, eps2) in &[
        (0.05_f32, 0.3_f32),
        (0.05, 0.5),
        (0.1, 0.5),
        (0.1, 0.7),
        (0.2, 0.8),
    ] {
        for &(d1, d2, d3) in &[
            (4usize, 7, 10),
            (5, 7, 10),
            (6, 8, 10),
            (6, 8, 11),
            (7, 9, 11),
        ] {
            let near = fit_plain_poly_for(ref_pow_sqrt, 0.0, eps1 as f64, d1);
            let mid = fit_plain_poly_for(ref_pow_sqrt, eps1 as f64, eps2 as f64, d2);
            let far = fit_plain_poly_for(ref_pow_sqrt, eps2 as f64, z_hi, d3);
            let m = measure_dense(
                |x| {
                    let z = x.sqrt();
                    if z < eps1 {
                        eval_poly_f32_horner(z, &near)
                    } else if z < eps2 {
                        eval_poly_f32_horner(z, &mid)
                    } else {
                        eval_poly_f32_horner(z, &far)
                    }
                },
                ref_pow,
                SWEEP_LO,
                SWEEP_HI,
                500_000,
            );
            println!("  eps=({eps1:>4.2},{eps2:>4.2}) deg=({d1},{d2},{d3}): {m}");
            if best_e
                .as_ref()
                .is_none_or(|(_, _, _, _, _, b)| m.max_abs < b.max_abs)
            {
                best_e = Some((eps1, eps2, near, mid, far, m));
            }
        }
    }

    // -------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------
    println!("\n===========================================================");
    println!("  Summary");
    println!("===========================================================");
    if let Some((deg, _, m)) = best_a.as_ref() {
        println!(
            "A (monomial √x):      deg {deg:>2}  max_abs={:.3e}",
            m.max_abs
        );
    }
    if let Some(b) = best_b.as_ref() {
        let m = measure_dense(
            |x| eval_f32_horner(x.sqrt(), &b.numerator, &b.denominator),
            ref_pow,
            SWEEP_LO,
            SWEEP_HI,
            1_000_000,
        );
        println!(
            "B (rational √x):     P{}/Q{}  max_abs={:.3e}",
            b.numerator.len() - 1,
            b.denominator.len() - 1,
            m.max_abs
        );
    }
    if let Some((deg, eps, _, _, m)) = best_d.as_ref() {
        println!(
            "D (piecewise √x):   eps={eps:.3} bulk_deg={deg:>2}  max_abs={:.3e}",
            m.max_abs
        );
    }
    if let Some((eps1, eps2, _, _, _, m)) = best_e.as_ref() {
        println!(
            "E (three-piece):   eps=({eps1:.2},{eps2:.2})        max_abs={:.3e}",
            m.max_abs
        );
    }

    // Emit the winner's coefficients.
    println!("\n=== Winner coefficients ===");
    let a_err = best_a
        .as_ref()
        .map(|(_, _, m)| m.max_abs)
        .unwrap_or(f64::INFINITY);
    let b_err = if let Some(b) = best_b.as_ref() {
        let m = measure_dense(
            |x| eval_f32_horner(x.sqrt(), &b.numerator, &b.denominator),
            ref_pow,
            SWEEP_LO,
            SWEEP_HI,
            1_000_000,
        );
        m.max_abs
    } else {
        f64::INFINITY
    };
    let d_err = best_d
        .as_ref()
        .map(|(_, _, _, _, m)| m.max_abs)
        .unwrap_or(f64::INFINITY);

    let e_err = best_e
        .as_ref()
        .map(|(_, _, _, _, _, m)| m.max_abs)
        .unwrap_or(f64::INFINITY);
    // We DELIBERATELY prefer the 2-piece (D) over the 3-piece (E) when D
    // is "good enough" (≤ 5e-4 max abs, the SIMD parity tolerance). The
    // 3-piece's extra precision is excess — it costs an extra Horner chain
    // per channel for accuracy we don't need.
    let tolerance = 5e-4_f64;
    let winner;
    let winner_err;
    if d_err <= tolerance {
        winner = "D";
        winner_err = d_err;
    } else if e_err <= tolerance {
        winner = "E";
        winner_err = e_err;
    } else if b_err <= tolerance {
        winner = "B";
        winner_err = b_err;
    } else if a_err <= tolerance {
        winner = "A";
        winner_err = a_err;
    } else {
        // None hit tolerance — pick the lowest-error one and report.
        if e_err < d_err && e_err < a_err && e_err < b_err {
            winner = "E";
            winner_err = e_err;
        } else if d_err < a_err && d_err < b_err {
            winner = "D";
            winner_err = d_err;
        } else if a_err < b_err {
            winner = "A";
            winner_err = a_err;
        } else {
            winner = "B";
            winner_err = b_err;
        }
        println!("WARNING: no candidate hit {tolerance:.1e} max abs tolerance");
    }
    println!("Best: {winner} with max_abs={winner_err:.3e}");

    match winner {
        "A" => {
            if let Some((deg, coeffs, m)) = best_a {
                print_const(
                    "POW_INV24_SQRT_C",
                    &format!(
                        "x^(1/2.4) via monomial of √x, deg {}, max abs {:.3e} on [{}, {}].",
                        deg, m.max_abs, SWEEP_LO, SWEEP_HI,
                    ),
                    &coeffs,
                );
            }
        }
        "B" => {
            if let Some(b) = best_b {
                let m = measure_dense(
                    |x| eval_f32_horner(x.sqrt(), &b.numerator, &b.denominator),
                    ref_pow,
                    SWEEP_LO,
                    SWEEP_HI,
                    1_000_000,
                );
                print_const(
                    "POW_INV24_SQRT_P",
                    &format!(
                        "x^(1/2.4) via rational P(√x)/Q(√x), P deg {}, max abs {:.3e}.",
                        b.numerator.len() - 1,
                        m.max_abs,
                    ),
                    &b.numerator,
                );
                print_const(
                    "POW_INV24_SQRT_Q",
                    &format!(
                        "x^(1/2.4) rational denominator, Q deg {}.",
                        b.denominator.len() - 1,
                    ),
                    &b.denominator,
                );
            }
        }
        "D" => {
            if let Some((deg, eps, near, bulk, m)) = best_d {
                println!("const POW_INV24_SQRT_EPS: f32 = {eps:.6e};");
                print_const(
                    "POW_INV24_SQRT_NEAR",
                    &format!(
                        "Near-zero (z < {eps:.3}) branch, max abs {:.3e} overall.",
                        m.max_abs,
                    ),
                    &near,
                );
                print_const(
                    "POW_INV24_SQRT_BULK",
                    &format!(
                        "Bulk (z ≥ {eps:.3}) branch, deg {deg}, max abs {:.3e} overall.",
                        m.max_abs,
                    ),
                    &bulk,
                );
            }
        }
        "E" => {
            if let Some((eps1, eps2, near, mid, far, m)) = best_e {
                println!("const POW_INV24_EPS1: f32 = {eps1:.6e};");
                println!("const POW_INV24_EPS2: f32 = {eps2:.6e};");
                print_const(
                    "POW_INV24_NEAR",
                    &format!("Near (z < {eps1:.3}). Max abs {:.3e} overall.", m.max_abs),
                    &near,
                );
                print_const(
                    "POW_INV24_MID",
                    &format!(
                        "Mid ({eps1:.3} ≤ z < {eps2:.3}). Max abs {:.3e} overall.",
                        m.max_abs
                    ),
                    &mid,
                );
                print_const(
                    "POW_INV24_FAR",
                    &format!("Far (z ≥ {eps2:.3}). Max abs {:.3e} overall.", m.max_abs),
                    &far,
                );
            }
        }
        _ => unreachable!(),
    }
}
