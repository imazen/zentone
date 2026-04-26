// libultrahdr_gamut.cpp
//
// Standalone extraction of libultrahdr's six gamut-conversion functions
// (`bt709ToP3`, `bt709ToBt2100`, `p3ToBt709`, `p3ToBt2100`, `bt2100ToBt709`,
// `bt2100ToP3`) for comparison against zentone's matrix constants in
// `gamut.rs` (`BT709_TO_P3`, `BT709_TO_BT2020`, etc).
//
// Source file : lib/src/gainmapmath.cpp (lines 605-633)
// Upstream    : https://github.com/google/libultrahdr
// Commit      : 8cbc983
// License     : Apache-2.0
//
// Documented divergence: libultrahdr's matrices are quoted to six decimal
// places; zentone's are quoted to four. The parity test must use a 5e-3
// tolerance to absorb that. Both matrices derive from the same chromaticity
// + D65 conversion, just rounded differently.

#include <array>
#include <cmath>
#include <cstdio>

struct Color {
    float r, g, b;
};

// ---------- verbatim from libultrahdr lib/src/gainmapmath.cpp ----------

const std::array<float, 9> kBt709ToP3 = {
    0.822462f,  0.177537f, 0.000001f,
    0.033194f,  0.966807f, -0.000001f,
    0.017083f,  0.072398f, 0.91052f,
};
const std::array<float, 9> kBt709ToBt2100 = {
    0.627404f, 0.329282f, 0.043314f,
    0.069097f, 0.919541f, 0.011362f,
    0.016392f, 0.088013f, 0.895595f,
};
const std::array<float, 9> kP3ToBt709 = {
    1.22494f,   -0.22494f,  0.0f,
    -0.042057f, 1.042057f,  0.0f,
    -0.019638f, -0.078636f, 1.098274f,
};
const std::array<float, 9> kP3ToBt2100 = {
    0.753833f, 0.198597f, 0.04757f,
    0.045744f, 0.941777f, 0.012479f,
    -0.00121f, 0.017601f, 0.983608f,
};
const std::array<float, 9> kBt2100ToBt709 = {
    1.660491f,  -0.587641f, -0.07285f,
    -0.124551f, 1.1329f,    -0.008349f,
    -0.018151f, -0.100579f, 1.11873f,
};
const std::array<float, 9> kBt2100ToP3 = {
    1.343578f,  -0.282179f, -0.061399f,
    -0.065298f, 1.075788f,  -0.01049f,
    0.002822f,  -0.019598f, 1.016777f,
};

static Color ConvertGamut(Color e, const std::array<float, 9>& c) {
    return {
        c[0] * e.r + c[1] * e.g + c[2] * e.b,
        c[3] * e.r + c[4] * e.g + c[5] * e.b,
        c[6] * e.r + c[7] * e.g + c[8] * e.b,
    };
}

static Color bt709ToP3(Color e) { return ConvertGamut(e, kBt709ToP3); }
static Color bt709ToBt2100(Color e) { return ConvertGamut(e, kBt709ToBt2100); }
static Color p3ToBt709(Color e) { return ConvertGamut(e, kP3ToBt709); }
static Color p3ToBt2100(Color e) { return ConvertGamut(e, kP3ToBt2100); }
static Color bt2100ToBt709(Color e) { return ConvertGamut(e, kBt2100ToBt709); }
static Color bt2100ToP3(Color e) { return ConvertGamut(e, kBt2100ToP3); }

// --------------------------------- harness ---------------------------------

static void emit(const char* tag, Color in, Color out) {
    std::printf("%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", tag,
                (double)in.r, (double)in.g, (double)in.b,
                (double)out.r, (double)out.g, (double)out.b);
}

int main() {
    std::puts("# libultrahdr gamut-conversion matrices");
    std::puts("# columns: conv, r_in, g_in, b_in, r_out, g_out, b_out");
    std::puts("# conv values: bt709_to_p3, bt709_to_bt2100, p3_to_bt709,");
    std::puts("#              p3_to_bt2100, bt2100_to_bt709, bt2100_to_p3");
    std::puts("# reference: lib/src/gainmapmath.cpp:609-633 @ libultrahdr commit 8cbc983");
    std::puts("conv,r_in,g_in,b_in,r_out,g_out,b_out");

    // Inputs: greyscale, primaries, secondaries, mixed in-gamut and
    // wide-gamut HDR values up to 4× white. Negative inputs are out of
    // scope for primary conversion.
    const float channel_levels[] = {
        0.0f, 0.001f, 0.01f, 0.05f, 0.1f, 0.18f, 0.25f, 0.5f,
        0.75f, 1.0f, 1.5f, 2.0f, 4.0f,
    };

    auto run_all = [](Color in) {
        emit("bt709_to_p3", in, bt709ToP3(in));
        emit("bt709_to_bt2100", in, bt709ToBt2100(in));
        emit("p3_to_bt709", in, p3ToBt709(in));
        emit("p3_to_bt2100", in, p3ToBt2100(in));
        emit("bt2100_to_bt709", in, bt2100ToBt709(in));
        emit("bt2100_to_p3", in, bt2100ToP3(in));
    };

    // Greys
    for (float v : channel_levels) run_all({v, v, v});
    // Primaries (R, G, B at each level)
    for (float v : channel_levels) {
        run_all({v, 0.0f, 0.0f});
        run_all({0.0f, v, 0.0f});
        run_all({0.0f, 0.0f, v});
    }
    // Secondaries: cyan/magenta/yellow at each level
    for (float v : channel_levels) {
        run_all({0.0f, v, v});
        run_all({v, 0.0f, v});
        run_all({v, v, 0.0f});
    }
    // Mixed
    const Color mixed[] = {
        {0.5f, 0.3f, 0.1f}, {0.1f, 0.7f, 0.4f}, {0.9f, 0.2f, 0.6f},
        {0.3f, 0.6f, 0.9f}, {0.95f, 0.05f, 0.5f}, {0.45f, 0.55f, 0.65f},
        {2.0f, 1.0f, 0.5f}, {3.5f, 0.2f, 1.8f}, {4.0f, 4.0f, 4.0f},
    };
    for (const Color& c : mixed) run_all(c);

    return 0;
}
