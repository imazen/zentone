// libplacebo_bt2390.cpp
//
// Standalone extraction of libplacebo's BT.2390 EETF tone mapper for
// golden-file comparison against zentone's Bt2408Tonemapper and
// bt2390_tonemap.
//
// Source file : src/tone_mapping.c (function bt2390)
// Upstream    : https://code.videolan.org/videolan/libplacebo
//               Mirror: https://github.com/haasn/libplacebo
// License     : LGPL-2.1+
//
// The libplacebo implementation operates on PQ-normalized values in
// [0, 1]. It takes input_min/input_max (PQ-encoded source black/white)
// and output_min/output_max (PQ-encoded target black/white), normalizes
// to the source range, applies the Hermite knee + black point
// adaptation, then denormalizes back.
//
// This is the canonical BT.2390 EETF used by VLC (via libplacebo) and
// mpv. The Hermite spline matches zentone's Bt2408Tonemapper and
// bt2390_tonemap exactly per the agent research.
//
// Differences from zentone:
//   - libplacebo: ks = (1 + offset) * maxLum - offset, offset defaults 1.0
//     At offset=0.5: ks = 1.5*maxLum - 0.5 (ITU standard, matches zentone)
//   - libplacebo: bp = min(1/minLum, 4) (dynamic exponent)
//     zentone: fixed (1-E2)^4 (equivalent when minLum >= 0.25)
//   - libplacebo: gain correction for black point
//     zentone: no gain correction

#include <algorithm>
#include <cmath>
#include <cstdio>

// ---- PQ transfer function (ST.2084) ----

static const float PQ_M1 = 2610.0f / 16384.0f;
static const float PQ_M2 = 2523.0f / 4096.0f * 128.0f;
static const float PQ_C1 = 3424.0f / 4096.0f;
static const float PQ_C2 = 2413.0f / 4096.0f * 32.0f;
static const float PQ_C3 = 2392.0f / 4096.0f * 32.0f;

static float pq_oetf(float x) {
    // Linear [0,1] (normalized to 10000 nits) -> PQ [0,1]
    if (x <= 0.0f) return 0.0f;
    float xm1 = std::pow(x, PQ_M1);
    return std::pow((PQ_C1 + PQ_C2 * xm1) / (1.0f + PQ_C3 * xm1), PQ_M2);
}

static float pq_eotf(float x) {
    // PQ [0,1] -> Linear [0,1] (normalized to 10000 nits)
    if (x <= 0.0f) return 0.0f;
    float xm2 = std::pow(x, 1.0f / PQ_M2);
    float num = std::fmax(xm2 - PQ_C1, 0.0f);
    float den = PQ_C2 - PQ_C3 * xm2;
    if (den <= 0.0f) return 0.0f;
    return std::pow(num / den, 1.0f / PQ_M1);
}

// ---- BT.2390 EETF from libplacebo ----

// Operates in PQ-normalized domain.
// input_min/max: PQ-encoded source display black/white
// output_min/max: PQ-encoded target display black/white
// knee_offset: libplacebo parameter (ITU standard = 0.5)
struct Bt2390Params {
    float input_min;
    float input_max;
    float output_min;
    float output_max;
    float knee_offset;
};

static float bt2390_eetf(float x, const Bt2390Params& p) {
    // Normalize to source range
    float range = p.input_max - p.input_min;
    if (range <= 0.0f) return x;
    float xn = (x - p.input_min) / range;

    // Target display range in normalized coords
    float minLum = (p.output_min - p.input_min) / range;
    float maxLum = (p.output_max - p.input_min) / range;

    float ks = (1.0f + p.knee_offset) * maxLum - p.knee_offset;
    float bp = minLum > 0.0f ? std::fmin(1.0f / minLum, 4.0f) : 4.0f;
    float gain_inv = 1.0f + minLum / maxLum * std::pow(1.0f - maxLum, bp);
    float gain = maxLum < 1.0f ? 1.0f / gain_inv : 1.0f;

    // Hermite spline
    if (ks < 1.0f && xn >= ks) {
        float tb = (xn - ks) / (1.0f - ks);
        float tb2 = tb * tb;
        float tb3 = tb2 * tb;
        float pb = (2.0f * tb3 - 3.0f * tb2 + 1.0f) * ks
                 + (tb3 - 2.0f * tb2 + tb) * (1.0f - ks)
                 + (-2.0f * tb3 + 3.0f * tb2) * maxLum;
        xn = pb;
    }

    // Black point adaptation
    if (xn < 1.0f) {
        xn += minLum * std::pow(1.0f - xn, bp);
        xn = gain * (xn - minLum) + minLum;
    }

    // Denormalize back to PQ range
    return xn * range + p.input_min;
}

// ---- Convenience: nits-based wrapper ----

// Takes source content peak nits, target display peak nits, and a
// linear-light input value (normalized to 10000 nits). Returns
// linear-light output.
static float bt2390_nits(float linear_in, float content_nits,
                         float display_nits, float knee_offset) {
    float pq_in = pq_oetf(linear_in);

    Bt2390Params p;
    p.input_min = pq_oetf(0.0f);
    p.input_max = pq_oetf(content_nits / 10000.0f);
    p.output_min = pq_oetf(0.0f);
    p.output_max = pq_oetf(display_nits / 10000.0f);
    p.knee_offset = knee_offset;

    float pq_out = bt2390_eetf(pq_in, p);
    return pq_eotf(pq_out);
}

// ---- Scene-linear BT.2390 (matching zentone's bt2390_tonemap) ----

// Operates in scene-linear normalized domain, NOT PQ domain.
// This matches zentone's curves::bt2390_tonemap_ext.
static float bt2390_scene_linear(float e1, float source_peak,
                                  float target_peak, float min_lum_opt,
                                  bool use_min_lum) {
    if (source_peak <= target_peak) return e1;

    float ks = std::clamp(1.5f * target_peak / source_peak - 0.5f, 0.0f, 1.0f);

    float e2;
    if (e1 < ks) {
        e2 = e1;
    } else {
        float t = (e1 - ks) / (1.0f - ks);
        float t2 = t * t;
        float t3 = t2 * t;
        float p0 = ks;
        float p1 = 1.0f;
        float m0 = 1.0f - ks;
        float m1 = 0.0f;
        e2 = (2*t3 - 3*t2 + 1)*p0 + (t3 - 2*t2 + t)*m0
           + (-2*t3 + 3*t2)*p1 + (t3 - t2)*m1;
    }

    float e3;
    if (use_min_lum) {
        float om = 1.0f - e2;
        float om2 = om * om;
        e3 = min_lum_opt * (om2 * om2) + e2;
    } else {
        e3 = e2;
    }

    return e3 * target_peak / source_peak;
}

// --------------------------------- harness ---------------------------------

int main() {
    // === Section 1: PQ-domain BT.2390 (libplacebo style) ===
    // This is what zentone's Bt2408Tonemapper should match.
    std::puts("# libplacebo BT.2390 EETF (PQ domain)");
    std::puts("# content_nits, display_nits, knee_offset, input_nits, output_linear");
    std::puts("content_nits,display_nits,knee_offset,input_nits,output_linear");

    struct NitsConfig {
        float content;
        float display;
        float offset;
    };
    const NitsConfig configs[] = {
        {4000.0f, 1000.0f, 0.5f},  // ITU standard offset
        {4000.0f, 1000.0f, 1.0f},  // libplacebo default offset
        {10000.0f, 203.0f, 0.5f},  // PQ full range to SDR
        {1000.0f, 400.0f, 0.5f},   // moderate compression
        {4000.0f, 4000.0f, 0.5f},  // no compression (passthrough)
    };
    const float input_nits[] = {
        0.0f, 0.01f, 0.1f, 1.0f, 5.0f, 10.0f, 50.0f, 100.0f,
        203.0f, 400.0f, 1000.0f, 2000.0f, 4000.0f, 10000.0f,
    };

    for (const auto& cfg : configs) {
        for (float nits : input_nits) {
            if (nits > cfg.content) continue;
            float linear_in = nits / 10000.0f;
            float linear_out = bt2390_nits(linear_in, cfg.content,
                                           cfg.display, cfg.offset);
            float out_nits = linear_out * 10000.0f;
            std::printf("%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        (double)cfg.content, (double)cfg.display,
                        (double)cfg.offset, (double)nits, (double)out_nits);
        }
    }

    // === Section 2: Scene-linear BT.2390 (matching zentone bt2390_tonemap) ===
    std::puts("");
    std::puts("# Scene-linear BT.2390 EETF");
    std::puts("# source_peak, target_peak, min_lum, input, output");
    std::puts("source_peak,target_peak,min_lum,input,output");

    struct LinearConfig {
        float source;
        float target;
        float min_lum;
        bool use_min;
    };
    const LinearConfig lin_configs[] = {
        {1000.0f, 100.0f, 0.0f, false},
        {1000.0f, 100.0f, 0.05f, true},
        {4.0f, 1.0f, 0.0f, false},
        {8.0f, 1.0f, 0.0f, false},
        {100.0f, 100.0f, 0.0f, false},  // passthrough
    };
    const float lin_inputs[] = {
        0.0f, 0.001f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f,
        0.75f, 0.9f, 0.95f, 0.99f, 1.0f,
    };

    for (const auto& cfg : lin_configs) {
        for (float x : lin_inputs) {
            float out = bt2390_scene_linear(x, cfg.source, cfg.target,
                                            cfg.min_lum, cfg.use_min);
            std::printf("%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        (double)cfg.source, (double)cfg.target,
                        (double)cfg.min_lum, (double)x, (double)out);
        }
    }

    return 0;
}
