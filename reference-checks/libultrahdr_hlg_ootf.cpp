// libultrahdr_hlg_ootf.cpp
//
// Standalone extraction of libultrahdr's HLG OOTF and inverse OOTF for
// comparison against zentone's `hlg::hlg_ootf` / `hlg::hlg_inverse_ootf`.
//
// Source file : lib/src/gainmapmath.cpp (lines 289-312)
// Upstream    : https://github.com/google/libultrahdr
// Commit      : 8cbc983
// License     : Apache-2.0
//
// libultrahdr exposes `hlgOotf(Color, LuminanceFn)` and `hlgInverseOotf` with
// a hardcoded `kOotfGamma = 1.2`. The HDR luminance is computed via a
// caller-supplied `LuminanceFn` — for the BT.2100 path that's
// `bt2100Luminance` (kBt2100R/G/B = 0.2627 / 0.677998 / 0.059302).
//
// zentone's `hlg_ootf(rgb, gamma)` accepts the gamma directly (so we can
// also exercise `hlg_system_gamma(display_peak_nits)` values like 1.2 for
// 1000 nits, 1.453 for 4000 nits) and uses LR/LG/LB = 0.2627/0.6780/0.0593
// internally. The luminance precision difference (~1e-4 absolute) is the
// only documented divergence; the OOTF itself is bit-identical math.

#include <array>
#include <cmath>
#include <cstdio>

struct Color {
    float r, g, b;
};

// ---------- verbatim from libultrahdr lib/src/gainmapmath.cpp ----------

static const float kBt2100R = 0.2627f, kBt2100G = 0.677998f, kBt2100B = 0.059302f;
static const float kOotfGamma = 1.2f;

static float bt2100Luminance(Color e) {
    return kBt2100R * e.r + kBt2100G * e.g + kBt2100B * e.b;
}

static Color hlgOotf(Color e, float (*luminance)(Color), float gamma) {
    float y = luminance(e);
    float scale = std::pow(y, gamma - 1.0f);
    return {e.r * scale, e.g * scale, e.b * scale};
}

static Color hlgInverseOotf(Color e, float (*luminance)(Color), float gamma) {
    float y = luminance(e);
    float scale = std::pow(y, (1.0f / gamma) - 1.0f);
    return {e.r * scale, e.g * scale, e.b * scale};
}

// --------------------------------- harness ---------------------------------

int main() {
    std::puts("# libultrahdr HLG OOTF / inverse OOTF (BT.2100 luminance)");
    std::puts("# columns: dir, gamma, r_in, g_in, b_in, r_out, g_out, b_out");
    std::puts("# dir values: ootf, inverse_ootf");
    std::puts("# gamma values cover hlg_system_gamma over typical display peaks");
    std::puts("# reference: lib/src/gainmapmath.cpp:294/304 @ libultrahdr commit 8cbc983");
    std::puts("dir,gamma,r_in,g_in,b_in,r_out,g_out,b_out");

    // gamma values: 1.0 (identity OOTF), 1.2 (BT.2100 reference, 1000 nits),
    // 1.033 (~400 nits), 1.453 (~4000 nits), 1.5 (boundary).
    const float gammas[] = {1.0f, 1.033f, 1.2f, 1.453f, 1.5f};

    // Inputs: skip pure black (luminance==0 means pow(0, x) where x<0 is
    // undefined; libultrahdr only calls these on linearized scene-referred
    // light where Y > 0 in practice).
    const float ch_grid[] = {
        0.001f, 0.01f, 0.05f, 0.1f, 0.18f, 0.25f, 0.5f, 0.75f, 1.0f,
    };

    auto run = [&](float gamma, Color in) {
        Color fwd = hlgOotf(in, bt2100Luminance, gamma);
        Color inv = hlgInverseOotf(in, bt2100Luminance, gamma);
        std::printf("ootf,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    (double)gamma,
                    (double)in.r, (double)in.g, (double)in.b,
                    (double)fwd.r, (double)fwd.g, (double)fwd.b);
        std::printf("inverse_ootf,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    (double)gamma,
                    (double)in.r, (double)in.g, (double)in.b,
                    (double)inv.r, (double)inv.g, (double)inv.b);
    };

    // Greys
    for (float g : gammas) {
        for (float v : ch_grid) run(g, {v, v, v});
    }
    // Primaries (each axis at each level)
    for (float g : gammas) {
        for (float v : ch_grid) {
            run(g, {v, 0.001f, 0.001f});
            run(g, {0.001f, v, 0.001f});
            run(g, {0.001f, 0.001f, v});
        }
    }
    // Mixed: skinlike, sky, foliage, peak white edge cases
    const Color mixed[] = {
        {0.65f, 0.45f, 0.35f},  // skin-tone-ish
        {0.30f, 0.55f, 0.85f},  // sky
        {0.20f, 0.55f, 0.15f},  // foliage
        {1.0f,  0.5f,  0.25f},
        {0.85f, 0.10f, 0.10f},
        {0.10f, 0.85f, 0.10f},
        {0.10f, 0.10f, 0.85f},
        {0.99f, 0.99f, 0.99f},
    };
    for (float g : gammas) {
        for (const Color& c : mixed) run(g, c);
    }

    return 0;
}
