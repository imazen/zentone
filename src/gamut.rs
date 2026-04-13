//! Color gamut conversion and soft clipping.
//!
//! 3×3 linear-light RGB-to-RGB matrices for BT.709, Display P3, and
//! BT.2020, plus a hue-preserving [`soft_clip`] for out-of-gamut colors.
//!
//! When converting from a wider gamut to a narrower one (e.g., BT.2020 →
//! BT.709), some colors land outside `[0, 1]`. A hard per-channel clamp
//! shifts hue; [`soft_clip`] preserves the channel ratio so hue stays
//! constant. Use [`apply_matrix_clip`] for the combined operation.
//!
//! All matrices are row-major: `out[i] = sum(M[i][j] * in[j])`.
//! Derived from the CIE 1931 xy chromaticities of each primaries set
//! through the standard D65 white point.

/// Convert linear BT.2020 RGB to linear BT.709 RGB.
///
/// Clipping may be needed after conversion — BT.2020 gamut is wider
/// than BT.709, so some colors produce negative BT.709 values.
pub const BT2020_TO_BT709: [[f32; 3]; 3] = [
    [1.6605, -0.5876, -0.0728],
    [-0.1246, 1.1329, -0.0083],
    [-0.0182, -0.1006, 1.1187],
];

/// Convert linear BT.709 RGB to linear BT.2020 RGB.
pub const BT709_TO_BT2020: [[f32; 3]; 3] = [
    [0.6274, 0.3293, 0.0433],
    [0.0691, 0.9195, 0.0114],
    [0.0164, 0.0880, 0.8956],
];

/// Convert linear Display P3 RGB to linear BT.709 RGB.
pub const P3_TO_BT709: [[f32; 3]; 3] = [
    [1.2249, -0.2247, 0.0],
    [-0.0420, 1.0419, 0.0],
    [-0.0197, -0.0786, 1.0979],
];

/// Convert linear BT.709 RGB to linear Display P3 RGB.
pub const BT709_TO_P3: [[f32; 3]; 3] = [
    [0.8225, 0.1774, 0.0],
    [0.0332, 0.9669, 0.0],
    [0.0171, 0.0724, 0.9108],
];

/// Convert linear BT.2020 RGB to linear Display P3 RGB.
pub const BT2020_TO_P3: [[f32; 3]; 3] = [
    [1.3435, -0.2822, -0.0613],
    [-0.0653, 1.0758, -0.0105],
    [-0.0028, -0.0196, 1.0219],
];

/// Convert linear Display P3 RGB to linear BT.2020 RGB.
pub const P3_TO_BT2020: [[f32; 3]; 3] = [
    [0.7539, 0.1986, 0.0476],
    [0.0457, 0.9418, 0.0125],
    [0.0012, 0.0176, 0.9811],
];

/// Apply a 3×3 matrix to an RGB triple (row-major).
#[inline]
pub fn apply_matrix(m: &[[f32; 3]; 3], rgb: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2],
        m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2],
        m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2],
    ]
}

/// Apply a 3×3 matrix to a row of interleaved RGB f32 pixels in place.
pub fn apply_matrix_row(m: &[[f32; 3]; 3], row: &mut [f32], channels: usize) {
    debug_assert!(channels == 3 || channels == 4);
    for chunk in row.chunks_exact_mut(channels) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        let out = apply_matrix(m, rgb);
        chunk[0] = out[0];
        chunk[1] = out[1];
        chunk[2] = out[2];
    }
}

/// Returns `true` if any channel is outside `[0, 1]`.
#[inline]
pub fn is_out_of_gamut(rgb: [f32; 3]) -> bool {
    rgb[0] < 0.0 || rgb[0] > 1.0 || rgb[1] < 0.0 || rgb[1] > 1.0 || rgb[2] < 0.0 || rgb[2] > 1.0
}

/// Hue-preserving soft clip for out-of-gamut highlights.
///
/// Sorts channels by magnitude, clamps the max to 1.0, then linearly
/// interpolates the mid channel to preserve the ratio
/// `(mid - min) / (max - min)`. This keeps hue constant while pulling
/// over-range values back into `[0, 1]`.
///
/// Negative values (under-range from matrix conversion) are clamped to 0
/// separately. This handles the common case where BT.2020 → BT.709
/// produces small negative values on saturated colors.
#[inline]
pub fn soft_clip(rgb: [f32; 3]) -> [f32; 3] {
    let [mut r, mut g, mut b] = rgb;

    // Handle negatives first (under-range from gamut matrix).
    r = r.max(0.0);
    g = g.max(0.0);
    b = b.max(0.0);

    // If nothing exceeds 1.0, we're done.
    if r <= 1.0 && g <= 1.0 && b <= 1.0 {
        return [r, g, b];
    }

    // Sort into (max, mid, min) and apply hue-preserving clip.
    // The clip scales max to 1.0 and interpolates mid to keep the
    // ratio (mid - min) / (max - min) constant.
    if r >= g {
        if g > b {
            // r >= g > b
            clip_sorted(&mut r, &mut g, &mut b);
        } else if b > r {
            // b > r >= g
            clip_sorted(&mut b, &mut r, &mut g);
        } else if b > g {
            // r >= b > g
            clip_sorted(&mut r, &mut b, &mut g);
        } else {
            // r >= g == b (or g == b == 0 edge)
            r = r.min(1.0);
            g = g.min(1.0);
        }
    } else if r >= b {
        // g > r >= b
        clip_sorted(&mut g, &mut r, &mut b);
    } else if b > g {
        // b > g > r
        clip_sorted(&mut b, &mut g, &mut r);
    } else {
        // g >= b > r
        clip_sorted(&mut g, &mut b, &mut r);
    }

    [r, g, b]
}

/// Clip sorted channels (hi >= mid >= lo) to [0, 1] preserving hue.
#[inline(always)]
fn clip_sorted(hi: &mut f32, mid: &mut f32, lo: &mut f32) {
    let new_hi = hi.min(1.0);
    let new_lo = lo.min(1.0);
    if *hi != *lo {
        *mid = new_lo + (new_hi - new_lo) * (*mid - *lo) / (*hi - *lo);
    } else {
        *mid = new_hi;
    }
    *hi = new_hi;
    *lo = new_lo;
}

/// Apply a 3×3 gamut matrix and soft-clip out-of-gamut results.
///
/// Combines [`apply_matrix`] with [`soft_clip`] in one call. Use this
/// when converting from a wider to a narrower gamut (e.g., BT.2020 → BT.709).
#[inline]
pub fn apply_matrix_clip(m: &[[f32; 3]; 3], rgb: [f32; 3]) -> [f32; 3] {
    let out = apply_matrix(m, rgb);
    if is_out_of_gamut(out) {
        soft_clip(out)
    } else {
        out
    }
}

/// Apply a 3×3 gamut matrix with soft-clip to a row of interleaved pixels.
pub fn apply_matrix_clip_row(m: &[[f32; 3]; 3], row: &mut [f32], channels: usize) {
    debug_assert!(channels == 3 || channels == 4);
    for chunk in row.chunks_exact_mut(channels) {
        let rgb = apply_matrix(m, [chunk[0], chunk[1], chunk[2]]);
        let out = if is_out_of_gamut(rgb) {
            soft_clip(rgb)
        } else {
            rgb
        };
        chunk[0] = out[0];
        chunk[1] = out[1];
        chunk[2] = out[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bt709_bt2020_roundtrip() {
        let rgb = [0.5_f32, 0.3, 0.8];
        let bt2020 = apply_matrix(&BT709_TO_BT2020, rgb);
        let back = apply_matrix(&BT2020_TO_BT709, bt2020);
        for i in 0..3 {
            assert!(
                (back[i] - rgb[i]).abs() < 1e-3,
                "BT.709↔BT.2020 roundtrip[{i}]: {:.5} vs {:.5}",
                back[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn bt709_p3_roundtrip() {
        let rgb = [0.5_f32, 0.3, 0.8];
        let p3 = apply_matrix(&BT709_TO_P3, rgb);
        let back = apply_matrix(&P3_TO_BT709, p3);
        for i in 0..3 {
            assert!(
                (back[i] - rgb[i]).abs() < 1e-3,
                "BT.709↔P3 roundtrip[{i}]: {:.5} vs {:.5}",
                back[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn bt2020_p3_roundtrip() {
        let rgb = [0.5_f32, 0.3, 0.8];
        let p3 = apply_matrix(&BT2020_TO_P3, rgb);
        let back = apply_matrix(&P3_TO_BT2020, p3);
        for i in 0..3 {
            assert!(
                (back[i] - rgb[i]).abs() < 1e-3,
                "BT.2020↔P3 roundtrip[{i}]: {:.5} vs {:.5}",
                back[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn neutral_gray_preserved() {
        // Neutral gray should be unchanged by any gamut conversion
        // (D65 white point is shared).
        let gray = [0.5_f32, 0.5, 0.5];
        for (name, m) in [
            ("709→2020", &BT709_TO_BT2020),
            ("2020→709", &BT2020_TO_BT709),
            ("709→P3", &BT709_TO_P3),
            ("P3→709", &P3_TO_BT709),
        ] {
            let out = apply_matrix(m, gray);
            for (i, &c) in out.iter().enumerate() {
                assert!((c - 0.5).abs() < 0.01, "{name}: gray[{i}] = {c:.5}",);
            }
        }
    }

    #[test]
    fn row_preserves_alpha() {
        let mut row = [0.5_f32, 0.3, 0.8, 0.42];
        apply_matrix_row(&BT709_TO_BT2020, &mut row, 4);
        assert!((row[3] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn soft_clip_in_gamut_is_identity() {
        let rgb = [0.5, 0.3, 0.8];
        let clipped = soft_clip(rgb);
        for i in 0..3 {
            assert!(
                (clipped[i] - rgb[i]).abs() < 1e-7,
                "in-gamut soft_clip changed channel {i}"
            );
        }
    }

    #[test]
    fn soft_clip_clamps_to_unit_range() {
        // Saturated BT.2020 green → BT.709 produces out-of-gamut values.
        let bt2020_green = [0.0, 1.0, 0.0];
        let bt709 = apply_matrix(&BT2020_TO_BT709, bt2020_green);
        // Should have negative r and/or b
        assert!(
            is_out_of_gamut(bt709),
            "BT.2020 pure green should be out of BT.709 gamut"
        );

        let clipped = soft_clip(bt709);
        for (i, &c) in clipped.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&c),
                "soft_clip[{i}] = {c} out of [0,1]"
            );
        }
    }

    #[test]
    fn soft_clip_preserves_hue_better_than_hard_clamp() {
        // Over-range color: bright desaturated cyan-ish
        let rgb = [0.8, 1.3, 1.1];
        let clipped = soft_clip(rgb);
        let clamped = [
            rgb[0].clamp(0.0, 1.0),
            rgb[1].clamp(0.0, 1.0),
            rgb[2].clamp(0.0, 1.0),
        ];

        // Both should be in [0, 1]
        for &c in &clipped {
            assert!((0.0..=1.0).contains(&c));
        }

        // Soft clip should preserve the channel ordering (hue direction)
        // Original: g > b > r. Clipped should maintain g >= b >= r.
        assert!(clipped[1] >= clipped[2], "soft_clip: g should be >= b");
        assert!(clipped[2] >= clipped[0], "soft_clip: b should be >= r");

        // Hard clamp collapses g and b to 1.0, losing the relationship.
        // Soft clip should keep them different.
        assert!(
            (clipped[1] - clipped[2]).abs() > 0.01,
            "soft_clip should maintain g-b difference, got g={} b={}",
            clipped[1],
            clipped[2]
        );
        // Hard clamp: g=1.0, b=1.0 — difference is 0.
        assert!(
            (clamped[1] - clamped[2]).abs() < 1e-6,
            "hard clamp should collapse g and b"
        );
    }

    #[test]
    fn apply_matrix_clip_bt2020_saturated() {
        // Saturated BT.2020 colors should come out in [0,1] after clip.
        let colors = [
            [1.0_f32, 0.0, 0.0], // BT.2020 red
            [0.0, 1.0, 0.0],     // BT.2020 green
            [0.0, 0.0, 1.0],     // BT.2020 blue
            [1.0, 1.0, 0.0],     // BT.2020 yellow
            [0.0, 1.0, 1.0],     // BT.2020 cyan
        ];
        for color in &colors {
            let out = apply_matrix_clip(&BT2020_TO_BT709, *color);
            for (i, &c) in out.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&c),
                    "BT.2020 {color:?} → BT.709 clip[{i}] = {c}"
                );
            }
        }
    }
}
