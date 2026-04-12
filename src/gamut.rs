//! Color gamut conversion matrices for BT.709, Display P3, and BT.2020.
//!
//! These are the 3×3 linear-light RGB-to-RGB matrices used when the
//! EETF application space or the gain map's base/alternate colorimetry
//! differ from the output color space.
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
}
