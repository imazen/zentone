//! Verify the PQ and HLG transfer functions zentone depends on (via
//! `linear-srgb`) against the raw spec formulas computed independently
//! here.
//!
//! This is a belt-and-suspenders check: `linear-srgb` has its own tests,
//! but zentone's Bt2408Tonemapper, bt2390_tonemap, and every PQ-domain
//! operation depend on these being correct. If `linear-srgb` ever ships
//! a bad version, these tests catch it at the zentone level.
//!
//! Reference: ITU-R BT.2100 / SMPTE ST.2084 (PQ) and ARIB STD-B67 (HLG).

/// ST.2084 constants (exact rational forms from the spec).
const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

/// ARIB STD-B67 / BT.2100 HLG constants.
const HLG_A: f64 = 0.17883277;
const HLG_B: f64 = 0.28466892;
const HLG_C: f64 = 0.55991073;

/// Compute PQ OETF from the ST.2084 formula in f64 for maximum reference
/// precision. Input: linear [0,1] normalized to 10000 nits.
fn pq_oetf_f64(y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    let ym1 = y.powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * ym1) / (1.0 + PQ_C3 * ym1)).powf(PQ_M2)
}

/// Compute PQ EOTF from the ST.2084 formula in f64.
fn pq_eotf_f64(e: f64) -> f64 {
    if e <= 0.0 {
        return 0.0;
    }
    let em2 = e.powf(1.0 / PQ_M2);
    let num = (em2 - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * em2;
    if den <= 0.0 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1)
}

/// Compute HLG OETF from the BT.2100 formula in f64.
fn hlg_oetf_f64(e: f64) -> f64 {
    if e <= 0.0 {
        return 0.0;
    }
    if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        HLG_A * (12.0 * e - HLG_B).ln() + HLG_C
    }
}

/// Compute HLG inverse OETF in f64.
fn hlg_inv_oetf_f64(v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    if v <= 0.5 {
        v * v / 3.0
    } else {
        ((v - HLG_C) / HLG_A).exp() / 12.0 + HLG_B / 12.0
    }
}

/// Tolerance for PQ comparison (f32 vs f64 reference). The PQ curve has
/// steep slopes near black and near peak, so f32 loses precision faster
/// than sRGB. 1e-5 is ~20× the f32 ULP at 1.0.
const PQ_TOL: f32 = 1e-5;

/// Tolerance for HLG (less steep than PQ, but the log/exp can diverge).
const HLG_TOL: f32 = 1e-5;

// ============================================================================
// PQ tests
// ============================================================================

#[test]
fn pq_oetf_matches_st2084_formula() {
    let test_nits: &[f32] = &[
        0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 203.0, 500.0, 1000.0,
        4000.0, 10000.0,
    ];
    for &nits in test_nits {
        let linear = nits / 10000.0;
        let zt = linear_srgb::tf::linear_to_pq(linear);
        let reference = pq_oetf_f64(linear as f64) as f32;
        let err = (zt - reference).abs();
        assert!(
            err < PQ_TOL,
            "PQ OETF at {nits} nits: linear-srgb={zt}, ST.2084 f64={reference}, err={err}"
        );
    }
}

#[test]
fn pq_eotf_matches_st2084_formula() {
    let test_pq: &[f32] = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.58, 0.6, 0.7, 0.8, 0.9, 1.0];
    for &pq in test_pq {
        let zt = linear_srgb::tf::pq_to_linear(pq);
        let reference = pq_eotf_f64(pq as f64) as f32;
        let err = (zt - reference).abs();
        assert!(
            err < PQ_TOL,
            "PQ EOTF at PQ={pq}: linear-srgb={zt}, ST.2084 f64={reference}, err={err}"
        );
    }
}

#[test]
fn pq_roundtrip_preserves_value() {
    for i in 0..=20 {
        let linear = i as f32 / 20.0;
        let pq = linear_srgb::tf::linear_to_pq(linear);
        let back = linear_srgb::tf::pq_to_linear(pq);
        let err = (back - linear).abs();
        assert!(
            err < PQ_TOL,
            "PQ roundtrip at linear={linear}: back={back}, err={err}"
        );
    }
}

#[test]
fn pq_known_reference_points() {
    // SDR reference white: 203 nits = PQ ~0.58
    let sdr_pq = linear_srgb::tf::linear_to_pq(203.0 / 10000.0);
    assert!(
        (sdr_pq - 0.58).abs() < 0.01,
        "SDR white (203 nits) should be PQ ~0.58, got {sdr_pq}"
    );

    // Peak: 10000 nits = PQ 1.0
    let peak_pq = linear_srgb::tf::linear_to_pq(1.0);
    assert!(
        (peak_pq - 1.0).abs() < 1e-5,
        "10000 nits should be PQ 1.0, got {peak_pq}"
    );

    // Black: 0 nits = PQ ~0 (not exactly 0 due to PQ_C1)
    let black_pq = linear_srgb::tf::linear_to_pq(0.0);
    assert!(
        black_pq.abs() < 0.001,
        "0 nits should be PQ ~0, got {black_pq}"
    );
}

// ============================================================================
// HLG tests
// ============================================================================

#[test]
fn hlg_oetf_matches_bt2100_formula() {
    let test_vals: &[f32] = &[0.0, 0.001, 0.01, 1.0 / 12.0, 0.1, 0.25, 0.5, 0.75, 1.0];
    for &v in test_vals {
        let zt = linear_srgb::tf::linear_to_hlg(v);
        let reference = hlg_oetf_f64(v as f64) as f32;
        let err = (zt - reference).abs();
        assert!(
            err < HLG_TOL,
            "HLG OETF at {v}: linear-srgb={zt}, BT.2100 f64={reference}, err={err}"
        );
    }
}

#[test]
fn hlg_inv_oetf_matches_bt2100_formula() {
    let test_vals: &[f32] = &[0.0, 0.1, 0.25, 0.5, 0.75, 1.0];
    for &v in test_vals {
        let zt = linear_srgb::tf::hlg_to_linear(v);
        let reference = hlg_inv_oetf_f64(v as f64) as f32;
        let err = (zt - reference).abs();
        assert!(
            err < HLG_TOL,
            "HLG inv OETF at {v}: linear-srgb={zt}, BT.2100 f64={reference}, err={err}"
        );
    }
}

#[test]
fn hlg_roundtrip_preserves_value() {
    for i in 0..=20 {
        let linear = i as f32 / 20.0;
        let hlg = linear_srgb::tf::linear_to_hlg(linear);
        let back = linear_srgb::tf::hlg_to_linear(hlg);
        let err = (back - linear).abs();
        assert!(
            err < HLG_TOL,
            "HLG roundtrip at linear={linear}: back={back}, err={err}"
        );
    }
}

// ============================================================================
// libm vs std consistency (when std feature is on)
// ============================================================================

#[cfg(feature = "std")]
#[test]
fn libm_matches_std_for_zentone_math() {
    // zentone uses libm for no_std compat. Verify the functions it calls
    // match the std versions within 1 ULP.
    let test_vals = [0.001f32, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0];

    for &x in &test_vals {
        // powf
        let std_pow = x.powf(2.4);
        let libm_pow = libm::powf(x, 2.4);
        assert!(
            (std_pow - libm_pow).abs() < 1e-6,
            "powf({x}, 2.4): std={std_pow}, libm={libm_pow}"
        );

        // log2
        let std_log2 = x.log2();
        let libm_log2 = libm::log2f(x);
        assert!(
            (std_log2 - libm_log2).abs() < 1e-6,
            "log2({x}): std={std_log2}, libm={libm_log2}"
        );

        // sqrt
        let std_sqrt = x.sqrt();
        let libm_sqrt = libm::sqrtf(x);
        assert!(
            (std_sqrt - libm_sqrt).abs() < 1e-7,
            "sqrt({x}): std={std_sqrt}, libm={libm_sqrt}"
        );

        // exp
        if x < 5.0 {
            // avoid overflow
            let std_exp = x.exp();
            let libm_exp = libm::expf(x);
            assert!(
                (std_exp - libm_exp).abs() / std_exp.abs().max(1e-10) < 1e-6,
                "exp({x}): std={std_exp}, libm={libm_exp}"
            );
        }
    }
}
