// libavif_apply_gain.cpp
//
// Standalone extraction of libavif's gain map apply core loop for
// cross-check against libultrahdr's applyGain and zentone's (future)
// ultrahdr-core gain apply.
//
// Source file : src/gainmap.c (lines 40-280)
// Upstream    : https://github.com/AOMediaCodec/libavif
// Commit      : 58b3459 (HEAD of ~/work/libavif as of 2026-04-11)
// License     : BSD-2-Clause
// Permalink   : https://github.com/AOMediaCodec/libavif/blob/main/src/gainmap.c
//
// IMPORTANT INTEROP NOTE
// ----------------------
// libavif stores `gainMapMin` and `gainMapMax` **already in log2 units**,
// matching the ISO 21496-1 field names `min_log2` and `max_log2` from
// §5.2.5. libultrahdr, by contrast, stores `min_content_boost` and
// `max_content_boost` **in linear gain units** (so 1.0 = no boost, 4.0 =
// 4× boost) and computes `log2(...)` inside `applyGain`. Numerically
// equivalent but not interchangeable in the metadata representation.
//
// When comparing zentone/ultrahdr-core to these two references, you must
// feed each its own convention:
//   libavif:     gainMapMin = 0.0, gainMapMax = 2.0        (log2 of 1..4)
//   libultrahdr: min_content_boost = 1.0, max_content_boost = 4.0
//
// Same physical gain. Different field layout. ISO 21496-1 uses the
// libavif (log2) convention.
//
// Also note: libavif's `lerp(a, b, w)` returns `(1 - w) * a + w * b`,
// while libultrahdr writes the equivalent `log2(min) * (1 - g) + log2(max) * g`
// inline. Mathematically identical; bit-exact modulo FMA reordering.

#include <algorithm>
#include <cmath>
#include <cstdio>

struct AvifGainMap {
    float gainMapMin[3];   // log2 units
    float gainMapMax[3];   // log2 units
    float gainMapGamma[3]; // gamma (1 / gammaInv at apply time)
    float baseOffset[3];
    float alternateOffset[3];
    float baseHdrHeadroom;      // log2 units
    float alternateHdrHeadroom; // log2 units
};

// ---------- verbatim from libavif src/gainmap.c:44-61 ----------

static float avifGetGainMapWeight(float hdrHeadroom, const AvifGainMap* gainMap) {
    const float baseHdrHeadroom = gainMap->baseHdrHeadroom;
    const float alternateHdrHeadroom = gainMap->alternateHdrHeadroom;
    if (baseHdrHeadroom == alternateHdrHeadroom) {
        return 0.0f;
    }
    const float w = std::clamp(
        (hdrHeadroom - baseHdrHeadroom) / (alternateHdrHeadroom - baseHdrHeadroom), 0.0f, 1.0f);
    return (alternateHdrHeadroom < baseHdrHeadroom) ? -w : w;
}

static inline float lerp(float a, float b, float w) {
    return (1.0f - w) * a + w * b;
}

// ---------- verbatim inner loop from libavif src/gainmap.c:254 ----------

struct Color {
    float r, g, b;
};

static Color avifApplyGainPixel(Color baseLinear, Color gainMapValue,
                                const AvifGainMap* gainMap, float weight) {
    const float gammaInv[3] = {1.0f / gainMap->gainMapGamma[0], 1.0f / gainMap->gainMapGamma[1],
                               1.0f / gainMap->gainMapGamma[2]};

    Color out{};
    for (int c = 0; c < 3; ++c) {
        float bl = (&baseLinear.r)[c];
        float gv = (&gainMapValue.r)[c];

        const float gainMapLog2 = lerp(gainMap->gainMapMin[c], gainMap->gainMapMax[c],
                                       std::pow(gv, gammaInv[c]));
        const float toneMappedLinear =
            (bl + gainMap->baseOffset[c]) * std::exp2(gainMapLog2 * weight) -
            gainMap->alternateOffset[c];

        (&out.r)[c] = toneMappedLinear;
    }
    return out;
}

// --------------------------------- harness ---------------------------------

int main() {
    AvifGainMap gm{};
    // Log2 units: 0..2 corresponds to linear boost 1..4 (matches libultrahdr
    // single-channel default at min_content_boost=1, max_content_boost=4).
    for (int i = 0; i < 3; ++i) {
        gm.gainMapMin[i] = 0.0f;
        gm.gainMapMax[i] = 2.0f;
        gm.gainMapGamma[i] = 1.0f;
        gm.baseOffset[i] = 1.0f / 64.0f;
        gm.alternateOffset[i] = 1.0f / 64.0f;
    }
    gm.baseHdrHeadroom = 0.0f;      // log2(1.0) — SDR primary
    gm.alternateHdrHeadroom = 2.0f; // log2(4.0) — HDR target

    // Weight test: given various display hdrHeadroom values, compute the
    // weight and verify both the direct weight result and the apply result.
    std::puts("# libavif avifGetGainMapWeight");
    std::puts("# base_headroom_log2,alt_headroom_log2,display_headroom_log2,weight");
    std::puts("base_headroom_log2,alt_headroom_log2,display_headroom_log2,weight");

    const float headrooms[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f};
    for (float d : headrooms) {
        float w = avifGetGainMapWeight(d, &gm);
        std::printf("%.9g,%.9g,%.9g,%.9g\n",
                    (double)gm.baseHdrHeadroom, (double)gm.alternateHdrHeadroom,
                    (double)d, (double)w);
    }

    // Per-pixel apply: single-channel gain value drives all three, base
    // varies, weight fixed at 1.0 for full HDR.
    std::puts("");
    std::puts("# libavif apply gain core, weight=1.0");
    std::puts("# metadata: gainMapMin=0 gainMapMax=2 (log2) gamma=1 offsets=1/64");
    std::puts("# columns: gain, base, out_r, out_g, out_b");
    std::puts("gain,base,out_r,out_g,out_b");
    const float gains[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    const float bases[] = {0.0f, 0.01f, 0.1f, 0.18f, 0.5f, 0.75f, 1.0f};
    for (float gv : gains) {
        for (float b : bases) {
            Color base{b, b, b};
            Color gain{gv, gv, gv};
            Color out = avifApplyGainPixel(base, gain, &gm, 1.0f);
            std::printf("%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        (double)gv, (double)b,
                        (double)out.r, (double)out.g, (double)out.b);
        }
    }

    // Weight varying
    std::puts("");
    std::puts("# libavif apply gain core, varying weight at fixed gain=0.5, base=0.5");
    std::puts("# columns: weight, out_r");
    std::puts("weight,out_r");
    const float weights[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (float w : weights) {
        Color base{0.5f, 0.5f, 0.5f};
        Color gain{0.5f, 0.5f, 0.5f};
        Color out = avifApplyGainPixel(base, gain, &gm, w);
        std::printf("%.9g,%.9g\n", (double)w, (double)out.r);
    }

    return 0;
}
