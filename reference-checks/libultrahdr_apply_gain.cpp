// libultrahdr_apply_gain.cpp
//
// Standalone extraction of libultrahdr's applyGain functions for
// comparison against zentone's gain-map application formula.
//
// Source file : lib/src/gainmapmath.cpp (lines 797-861)
// Upstream    : https://github.com/google/libultrahdr
// Commit      : 8cbc983
// License     : Apache-2.0
//
// The formula is per-channel, with the log-space blend between
// min_content_boost and max_content_boost:
//
//     gain_power = pow(gain_normalized, 1 / gamma)         // optional
//     log_boost  = log2(min_boost) * (1 - gain_power)
//                + log2(max_boost) * gain_power
//     factor     = exp2(log_boost * weight)
//     out        = (e + offset_sdr) * factor - offset_hdr
//
// where `gain_normalized` is the gain map's per-pixel value already in
// [0, 1]. `weight` is the display's HDR headroom weight, also in [0, 1];
// libultrahdr uses 1.0 when displaying at full headroom.

#include <array>
#include <cmath>
#include <cstdio>

struct Color {
    float r, g, b;
};

// Subset of uhdr_gainmap_metadata_ext_t that matters for applyGain.
// Per-channel arrays of length 3.
struct GainMapMetadata {
    float min_content_boost[3];
    float max_content_boost[3];
    float gamma[3];
    float offset_sdr[3];
    float offset_hdr[3];
};

// ---------- verbatim from libultrahdr lib/src/gainmapmath.cpp:797 ----------

// NOTE: kept for documentation parity with upstream but unused in the
// test harness (the weighted variant is called with weight = 1.0 for the
// equivalent behavior).
[[maybe_unused]] static Color applyGain(Color e, float gain,
                                        const GainMapMetadata* metadata) {
    float g = gain;
    if (metadata->gamma[0] != 1.0f) g = std::pow(g, 1.0f / metadata->gamma[0]);
    float logBoost = std::log2(metadata->min_content_boost[0]) * (1.0f - g) +
                     std::log2(metadata->max_content_boost[0]) * g;
    float gainFactor = std::exp2(logBoost);
    return {(e.r + metadata->offset_sdr[0]) * gainFactor - metadata->offset_hdr[0],
            (e.g + metadata->offset_sdr[0]) * gainFactor - metadata->offset_hdr[0],
            (e.b + metadata->offset_sdr[0]) * gainFactor - metadata->offset_hdr[0]};
}

static Color applyGainWeighted(Color e, float gain, const GainMapMetadata* metadata,
                               float gainmapWeight) {
    float g = gain;
    if (metadata->gamma[0] != 1.0f) g = std::pow(g, 1.0f / metadata->gamma[0]);
    float logBoost = std::log2(metadata->min_content_boost[0]) * (1.0f - g) +
                     std::log2(metadata->max_content_boost[0]) * g;
    float gainFactor = std::exp2(logBoost * gainmapWeight);
    return {(e.r + metadata->offset_sdr[0]) * gainFactor - metadata->offset_hdr[0],
            (e.g + metadata->offset_sdr[0]) * gainFactor - metadata->offset_hdr[0],
            (e.b + metadata->offset_sdr[0]) * gainFactor - metadata->offset_hdr[0]};
}

static Color applyGainPerChannel(Color e, Color gain, const GainMapMetadata* metadata,
                                 float gainmapWeight) {
    float gr = gain.r, gg = gain.g, gb = gain.b;
    if (metadata->gamma[0] != 1.0f) gr = std::pow(gr, 1.0f / metadata->gamma[0]);
    if (metadata->gamma[1] != 1.0f) gg = std::pow(gg, 1.0f / metadata->gamma[1]);
    if (metadata->gamma[2] != 1.0f) gb = std::pow(gb, 1.0f / metadata->gamma[2]);
    float logBoostR = std::log2(metadata->min_content_boost[0]) * (1.0f - gr) +
                      std::log2(metadata->max_content_boost[0]) * gr;
    float logBoostG = std::log2(metadata->min_content_boost[1]) * (1.0f - gg) +
                      std::log2(metadata->max_content_boost[1]) * gg;
    float logBoostB = std::log2(metadata->min_content_boost[2]) * (1.0f - gb) +
                      std::log2(metadata->max_content_boost[2]) * gb;
    float factorR = std::exp2(logBoostR * gainmapWeight);
    float factorG = std::exp2(logBoostG * gainmapWeight);
    float factorB = std::exp2(logBoostB * gainmapWeight);
    return {(e.r + metadata->offset_sdr[0]) * factorR - metadata->offset_hdr[0],
            (e.g + metadata->offset_sdr[1]) * factorG - metadata->offset_hdr[1],
            (e.b + metadata->offset_sdr[2]) * factorB - metadata->offset_hdr[2]};
}

// --------------------------------- harness ---------------------------------

int main() {
    // Default UltraHDR metadata: offsets 1/64, gamma 1.0, boost range 1..4
    GainMapMetadata md{};
    for (int i = 0; i < 3; ++i) {
        md.min_content_boost[i] = 1.0f;
        md.max_content_boost[i] = 4.0f;
        md.gamma[i] = 1.0f;
        md.offset_sdr[i] = 1.0f / 64.0f;
        md.offset_hdr[i] = 1.0f / 64.0f;
    }

    // Per-pixel gain values (normalized [0, 1]) and SDR base values.
    const float gains[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    const float bases[] = {0.0f, 0.01f, 0.1f, 0.18f, 0.5f, 0.75f, 1.0f};
    const float weights[] = {0.0f, 0.5f, 1.0f};

    std::puts("# libultrahdr applyGain single-channel, weighted, gamma=1.0");
    std::puts("# metadata: min_boost=1.0 max_boost=4.0 offset_sdr=1/64 offset_hdr=1/64");
    std::puts("# columns: weight, gain, base, out_r, out_g, out_b");
    std::puts("weight,gain,base,out_r,out_g,out_b");
    for (float w : weights) {
        for (float g : gains) {
            for (float b : bases) {
                Color e{b, b, b};
                Color out = applyGainWeighted(e, g, &md, w);
                std::printf("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                            (double)w, (double)g, (double)b,
                            (double)out.r, (double)out.g, (double)out.b);
            }
        }
    }

    // Per-channel gain vectors
    std::puts("");
    std::puts("# libultrahdr applyGain per-channel, weight=1.0, gamma=1.0");
    std::puts("# columns: gain_r, gain_g, gain_b, base_r, base_g, base_b, out_r, out_g, out_b");
    std::puts("gain_r,gain_g,gain_b,base_r,base_g,base_b,out_r,out_g,out_b");

    struct Triple {
        Color gain;
        Color base;
    };
    const Triple cases[] = {
        {{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}},
        {{1.0f, 1.0f, 1.0f}, {0.5f, 0.5f, 0.5f}},
        {{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}},
        {{1.0f, 0.5f, 0.0f}, {0.3f, 0.6f, 0.1f}},
        {{0.25f, 0.75f, 1.0f}, {0.7f, 0.2f, 0.9f}},
    };
    for (const auto& c : cases) {
        Color out = applyGainPerChannel(c.base, c.gain, &md, 1.0f);
        std::printf("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    (double)c.gain.r, (double)c.gain.g, (double)c.gain.b,
                    (double)c.base.r, (double)c.base.g, (double)c.base.b,
                    (double)out.r, (double)out.g, (double)out.b);
    }

    return 0;
}
