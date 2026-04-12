// darktable_filmic.cpp
//
// Standalone extraction of darktable's V3 rational filmic spline for
// golden-file comparison against zentone's CompiledFilmicSpline.
//
// Source file : src/iop/filmicrgb.c (lines 2735-3030, 936-994, 1014-1024)
// Upstream    : https://github.com/darktable-org/darktable
// Commit      : a193e2707f (master, 2026-04-11)
// License     : GPL-3.0
//
// This file is a derivative work of darktable, licensed under GPL-3.0.
// It is used strictly for verification — compiled separately, never
// linked into zentone's AGPL/commercial library.
//
// Extracted elements:
//   1. V3 rational spline coefficient computation (compute_spline)
//   2. filmic_spline evaluation (rational branch)
//   3. log_tonemapping_v2_1ch (log-space shaper)
//   4. filmic_desaturate_v2

#include <algorithm>
#include <cmath>
#include <cstdio>

static const float SAFETY_MARGIN = 0.01f;
static const float NORM_MIN = 1.52587890625e-05f; // 2^-16

struct FilmicParams {
    float output_power;
    float latitude;  // %
    float white_point_source;  // EV
    float black_point_source;  // EV (negative)
    float contrast;
    float black_point_target;  // %
    float grey_point_target;   // %
    float white_point_target;  // %
    float balance;  // [-50, 50]
    float saturation;  // [0, 100]
};

struct SplineCoeffs {
    float M1[3], M2[3], M3[3], M4[3];
    float latitude_min, latitude_max;
    float grey_source, black_source, dynamic_range;
    float sigma_toe, sigma_shoulder, saturation;
};

static void compute_spline(const FilmicParams& p, SplineCoeffs& s) {
    float hardness = p.output_power;
    float grey_display = std::pow(0.1845f, 1.0f / hardness);
    float latitude = std::clamp(p.latitude, 0.0f, 100.0f) / 100.0f;
    float white_source = p.white_point_source;
    float black_source = p.black_point_source;
    float dynamic_range = white_source - black_source;
    float grey_log = std::abs(black_source) / dynamic_range;
    float white_log = 1.0f;
    float black_log = 0.0f;
    float black_display = std::pow(
        std::clamp(p.black_point_target, 0.0f, p.grey_point_target) / 100.0f,
        1.0f / hardness);
    float white_display = std::pow(
        std::max(p.white_point_target, p.grey_point_target) / 100.0f,
        1.0f / hardness);
    float balance = std::clamp(p.balance, -50.0f, 50.0f) / 100.0f;
    float slope = p.contrast * dynamic_range / 8.0f;
    float min_contrast = 1.0f;
    float mc_white = (white_display - grey_display) / (white_log - grey_log);
    if (std::isfinite(mc_white)) min_contrast = std::max(min_contrast, mc_white);
    float mc_black = (grey_display - black_display) / (grey_log - black_log);
    if (std::isfinite(mc_black)) min_contrast = std::max(min_contrast, mc_black);
    min_contrast += SAFETY_MARGIN;
    float contrast = slope / (hardness * std::pow(grey_display, hardness - 1.0f));
    contrast = std::clamp(contrast, min_contrast, 100.0f);
    float linear_intercept = grey_display - contrast * grey_log;
    float xmin = (black_display + SAFETY_MARGIN * (white_display - black_display) - linear_intercept) / contrast;
    float xmax = (white_display - SAFETY_MARGIN * (white_display - black_display) - linear_intercept) / contrast;
    float toe_log = (1.0f - latitude) * grey_log + latitude * xmin;
    float shoulder_log = (1.0f - latitude) * grey_log + latitude * xmax;
    float balance_correction = balance > 0.0f
        ? 2.0f * balance * (shoulder_log - grey_log)
        : 2.0f * balance * (grey_log - toe_log);
    toe_log -= balance_correction;
    shoulder_log -= balance_correction;
    toe_log = std::max(toe_log, xmin);
    shoulder_log = std::min(shoulder_log, xmax);
    float toe_display = toe_log * contrast + linear_intercept;
    float shoulder_display = shoulder_log * contrast + linear_intercept;

    s.latitude_min = toe_log;
    s.latitude_max = shoulder_log;
    s.saturation = 2.0f * p.saturation / 100.0f + 1.0f;
    s.sigma_toe = (toe_log / 3.0f) * (toe_log / 3.0f);
    s.sigma_shoulder = ((1.0f - shoulder_log) / 3.0f) * ((1.0f - shoulder_log) / 3.0f);
    s.grey_source = 0.1845f;
    s.black_source = black_source;
    s.dynamic_range = dynamic_range;

    // Linear segment
    s.M2[2] = contrast;
    s.M1[2] = toe_display - s.M2[2] * toe_log;

    // Toe rational (darktable: x = P0[0] - P1[0], P0=toe, P1=black)
    {
        float x = toe_log - black_log;      // positive
        float y = toe_display - black_display;
        float g = contrast;
        float jx = (x * g / y + 1.0f); jx = std::max(jx * jx, 4.0f);
        float b = g / (2.0f * y) + (std::sqrt(jx - 4.0f) - 1.0f) / (2.0f * x);
        float c = y / g * (b * x * x + x) / (b * x * x + x - y / g);
        float a = c * g;
        s.M1[0] = a;
        s.M2[0] = b;
        s.M3[0] = c;
        s.M4[0] = toe_display;
    }

    // Shoulder rational (darktable: x = P1[0] - P0[0], P0=shoulder, P1=white)
    {
        float x = white_log - shoulder_log;   // positive (note: opposite order from toe)
        float y = white_display - shoulder_display;
        float g = contrast;
        float jx = (x * g / y + 1.0f); jx = std::max(jx * jx, 4.0f);
        float b = g / (2.0f * y) + (std::sqrt(jx - 4.0f) - 1.0f) / (2.0f * x);
        float c = y / g * (b * x * x + x) / (b * x * x + x - y / g);
        float a = c * g;
        s.M1[1] = a;
        s.M2[1] = b;
        s.M3[1] = c;
        s.M4[1] = shoulder_display;
    }
}

static float filmic_spline(float x, const SplineCoeffs& s) {
    if (x < s.latitude_min) {
        float xi = s.latitude_min - x;
        float rat = xi * (xi * s.M2[0] + 1.0f);
        return s.M4[0] - s.M1[0] * rat / (rat + s.M3[0]);
    } else if (x > s.latitude_max) {
        float xi = x - s.latitude_max;
        float rat = xi * (xi * s.M2[1] + 1.0f);
        return s.M4[1] + s.M1[1] * rat / (rat + s.M3[1]);
    } else {
        return s.M1[2] + x * s.M2[2];
    }
}

static float log_shaper(float x, float grey, float black, float dynamic_range) {
    return std::clamp((std::log2(std::max(x, NORM_MIN) / grey) - black) / dynamic_range, 0.0f, 1.0f);
}

static float desaturate_v2(float x, float saturation, float sigma_toe, float sigma_shoulder) {
    float radius_toe = x;
    float radius_shoulder = 1.0f - x;
    float sat2 = 0.5f / std::sqrt(saturation);
    float key_toe = std::exp(-radius_toe * radius_toe / sigma_toe * sat2);
    float key_shoulder = std::exp(-radius_shoulder * radius_shoulder / sigma_shoulder * sat2);
    return saturation - (key_toe + key_shoulder) * saturation;
}

// Full filmic pipeline for a single RGB pixel
static void filmic_rgb(const float rgb_in[3], float rgb_out[3],
                       const SplineCoeffs& s, const float luma_coeffs[3]) {
    // Norm (luminance)
    float norm = std::max(
        rgb_in[0] * luma_coeffs[0] + rgb_in[1] * luma_coeffs[1] + rgb_in[2] * luma_coeffs[2],
        NORM_MIN);
    float ratios[3] = {rgb_in[0] / norm, rgb_in[1] / norm, rgb_in[2] / norm};
    float min_ratio = std::min({ratios[0], ratios[1], ratios[2]});
    if (min_ratio < 0.0f) {
        ratios[0] -= min_ratio;
        ratios[1] -= min_ratio;
        ratios[2] -= min_ratio;
    }
    norm = log_shaper(norm, s.grey_source, s.black_source, s.dynamic_range);
    float desat = desaturate_v2(norm, s.saturation, s.sigma_toe, s.sigma_shoulder);
    float mapped = std::clamp(filmic_spline(norm, s), 0.0f, 1.0f);
    for (int c = 0; c < 3; c++) {
        rgb_out[c] = std::clamp(
            (ratios[c] + (1.0f - ratios[c]) * (1.0f - desat)) * mapped,
            0.0f, 1.0f);
    }
}

// --------------------------------- harness ---------------------------------

int main() {
    const float luma_bt709[3] = {0.2126f, 0.7152f, 0.0722f};

    struct Config {
        const char* name;
        FilmicParams params;
    };

    Config configs[] = {
        {"zentone_defaults", {1.0f, 33.0f, 3.0f, -8.0f, 1.18f, 0.01517634f, 18.45f, 100.0f, 0.0f, 0.0f}},
        {"darktable_defaults", {4.0f, 0.01f, 4.0f, -8.0f, 1.0f, 0.01517634f, 18.45f, 100.0f, 0.0f, 0.0f}},
        {"high_contrast", {1.0f, 33.0f, 3.0f, -8.0f, 2.5f, 0.01517634f, 18.45f, 100.0f, 0.0f, 0.0f}},
        {"with_saturation", {1.0f, 33.0f, 3.0f, -8.0f, 1.18f, 0.01517634f, 18.45f, 100.0f, 0.0f, 50.0f}},
    };

    std::puts("# darktable filmic V3 rational spline golden vectors");
    std::puts("# columns: config, r_in, g_in, b_in, r_out, g_out, b_out");
    std::puts("config,r_in,g_in,b_in,r_out,g_out,b_out");

    float test_rgbs[][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.001f, 0.001f, 0.001f},
        {0.01f, 0.01f, 0.01f},
        {0.1f, 0.1f, 0.1f},
        {0.1845f, 0.1845f, 0.1845f},  // mid-gray
        {0.5f, 0.5f, 0.5f},
        {1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f},
        {4.0f, 4.0f, 4.0f},
        {0.3f, 0.6f, 0.1f},  // colored
        {0.8f, 0.2f, 0.05f}, // warm
        {0.1f, 0.9f, 0.05f}, // green-heavy
    };

    for (const auto& cfg : configs) {
        SplineCoeffs s{};
        compute_spline(cfg.params, s);

        for (const auto& rgb : test_rgbs) {
            float out[3];
            filmic_rgb(rgb, out, s, luma_bt709);
            std::printf("%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        cfg.name,
                        (double)rgb[0], (double)rgb[1], (double)rgb[2],
                        (double)out[0], (double)out[1], (double)out[2]);
        }
    }

    // Scalar spline eval at specific x values
    std::puts("");
    std::puts("# Scalar spline evaluation");
    std::puts("# columns: config, x, spline_y");
    std::puts("config,x,spline_y");

    float test_x[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    for (const auto& cfg : configs) {
        SplineCoeffs s{};
        compute_spline(cfg.params, s);
        for (float x : test_x) {
            float y = filmic_spline(x, s);
            std::printf("%s,%.9g,%.9g\n", cfg.name, (double)x, (double)y);
        }
    }

    return 0;
}
