// libultrahdr_compute_gain.cpp
//
// Standalone extraction of libultrahdr's `computeGain` — the encode-side
// inverse of applyGain. Given an SDR and HDR luminance, produces the log2
// gain value that applyGain should recover to reconstruct hdr from sdr.
//
// Source file : lib/src/gainmapmath.cpp (lines 779-788)
// Upstream    : https://github.com/google/libultrahdr
// Commit      : 8cbc983
// License     : Apache-2.0
//
// The key detail: for very dark SDR pixels (< 2/255), libultrahdr clamps
// the gain to 2.3 to prevent blowout on re-quantization. This is a
// libultrahdr-specific rule, not in ISO 21496-1, and zentone's gain map
// encode path (in ultrahdr-core, not zentone itself) should match it for
// interop.

#include <algorithm>
#include <cmath>
#include <cstdio>

static const float kSdrOffset = 1e-7f;
static const float kHdrOffset = 1e-7f;

// ---------- verbatim from libultrahdr lib/src/gainmapmath.cpp:779-788 ----------

static float computeGain(float sdr, float hdr) {
    float gain = std::log2((hdr + kHdrOffset) / (sdr + kSdrOffset));
    if (sdr < 2.f / 255.0f) {
        gain = std::min(gain, 2.3f);
    }
    return gain;
}

// --------------------------------- harness ---------------------------------

int main() {
    // Sweep SDR and HDR luminance values across the interesting regions:
    // - deep shadow (<2/255, triggers the 2.3 clamp if HDR >> SDR)
    // - just above shadow threshold
    // - midtones
    // - whites and HDR highlights
    const float sdrs[] = {
        0.0f,
        1.0f / 512.0f,   // well below 2/255 (0.00196...)
        1.0f / 255.0f,   // 1 code
        2.0f / 255.0f,   // clamp boundary
        3.0f / 255.0f,   // just above clamp
        0.1f, 0.18f, 0.5f, 1.0f,
    };
    const float hdrs[] = {
        0.0f, 0.001f, 0.01f, 0.05f, 0.1f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f,
    };

    std::puts("# libultrahdr computeGain");
    std::puts("# columns: sdr, hdr, gain_log2");
    std::puts("# formula: log2((hdr+1e-7)/(sdr+1e-7)), clamped to 2.3 when sdr < 2/255");
    std::puts("sdr,hdr,gain_log2");
    for (float s : sdrs) {
        for (float h : hdrs) {
            float g = computeGain(s, h);
            std::printf("%.9g,%.9g,%.9g\n", (double)s, (double)h, (double)g);
        }
    }
    return 0;
}
