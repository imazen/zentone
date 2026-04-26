// libultrahdr_luminance.cpp
//
// Standalone extraction of libultrahdr's per-gamut luminance helpers
// (`srgbLuminance`, `p3Luminance`, `bt2100Luminance`) for comparison
// against zentone's `LUMA_BT709` / `LUMA_P3` / `LUMA_BT2020` constants.
//
// Source file : lib/src/gainmapmath.cpp (lines 92-195)
// Upstream    : https://github.com/google/libultrahdr
// Commit      : 8cbc983
// License     : Apache-2.0
//
// Note: libultrahdr's BT.2100 luminance uses kBt2100G=0.677998 and
// kBt2100B=0.059302 (six-decimal precision). zentone's LUMA_BT2020 uses
// 0.6780 / 0.0593 (four-decimal). The tolerance in the parity test must
// account for that documented divergence (~5e-4 absolute).

#include <cmath>
#include <cstdio>

struct Color {
    float r, g, b;
};

// ---------- verbatim from libultrahdr lib/src/gainmapmath.cpp ----------

static const float kSrgbR = 0.212639f, kSrgbG = 0.715169f, kSrgbB = 0.072192f;
static const float kP3R = 0.2289746f, kP3G = 0.6917385f, kP3B = 0.0792869f;
static const float kBt2100R = 0.2627f, kBt2100G = 0.677998f, kBt2100B = 0.059302f;

static float srgbLuminance(Color e) { return kSrgbR * e.r + kSrgbG * e.g + kSrgbB * e.b; }
static float p3Luminance(Color e) { return kP3R * e.r + kP3G * e.g + kP3B * e.b; }
static float bt2100Luminance(Color e) {
    return kBt2100R * e.r + kBt2100G * e.g + kBt2100B * e.b;
}

// --------------------------------- harness ---------------------------------

int main() {
    // Brute-force RGB sweep: per-channel grid plus diagonal greys, primaries,
    // peak HDR-ish values, and small/large mixed values. Each row is one
    // (r,g,b) triple emitted with all three luminance variants.
    const float ch_grid[] = {
        0.0f, 0.001f, 0.01f, 0.05f, 0.1f, 0.18f, 0.25f, 0.5f,
        0.75f, 1.0f, 1.5f, 2.0f, 4.0f, 8.0f,
    };

    std::puts("# libultrahdr per-gamut luminance dot products");
    std::puts("# columns: r, g, b, srgb_luma, p3_luma, bt2100_luma");
    std::puts("# reference: lib/src/gainmapmath.cpp:94/164/195 @ libultrahdr commit 8cbc983");
    std::puts("r,g,b,srgb_luma,p3_luma,bt2100_luma");

    // Greyscale diagonal: r=g=b
    for (float v : ch_grid) {
        Color e{v, v, v};
        std::printf("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    (double)v, (double)v, (double)v,
                    (double)srgbLuminance(e), (double)p3Luminance(e),
                    (double)bt2100Luminance(e));
    }

    // Primaries (R, G, B) at varying intensities
    for (float v : ch_grid) {
        for (int axis = 0; axis < 3; ++axis) {
            Color e{0.0f, 0.0f, 0.0f};
            (&e.r)[axis] = v;
            std::printf("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        (double)e.r, (double)e.g, (double)e.b,
                        (double)srgbLuminance(e), (double)p3Luminance(e),
                        (double)bt2100Luminance(e));
        }
    }

    // Cross terms: r/g/b each from grid, but skip pure-zero/equal cases
    // already covered above. Sweep enough to hit ~150 rows total.
    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j) {
            // Pick 3 representative b values per (r,g) pair to keep count manageable.
            float b_vals[] = {0.0f, ch_grid[(i + j) % 14], 1.0f};
            for (float b : b_vals) {
                Color e{ch_grid[i], ch_grid[j], b};
                if (e.r == e.g && e.g == e.b) continue; // grey diagonal already emitted
                std::printf("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                            (double)e.r, (double)e.g, (double)e.b,
                            (double)srgbLuminance(e), (double)p3Luminance(e),
                            (double)bt2100Luminance(e));
            }
        }
    }

    return 0;
}
