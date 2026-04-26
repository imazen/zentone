//! Thin math helpers that work in `no_std`.
//!
//! Rust's `f32` transcendental methods (`powf`, `ln`, `exp`, …) live in `std`.
//! zentone targets `no_std + alloc`, so we route through `libm` here — which
//! provides identical results to glibc's libm in both `std` and `no_std` builds.

#[inline]
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

#[inline]
pub(crate) fn log2f(x: f32) -> f32 {
    libm::log2f(x)
}

#[cfg(feature = "experimental")]
#[inline]
pub(crate) fn lnf(x: f32) -> f32 {
    libm::logf(x)
}

#[cfg(feature = "experimental")]
#[inline]
pub(crate) fn floorf(x: f32) -> f32 {
    libm::floorf(x)
}

#[cfg(feature = "experimental")]
#[inline]
pub(crate) fn roundf(x: f32) -> f32 {
    libm::roundf(x)
}

#[inline]
pub(crate) fn expf(x: f32) -> f32 {
    libm::expf(x)
}

#[inline]
pub(crate) fn exp2f(x: f32) -> f32 {
    libm::exp2f(x)
}

#[inline]
pub(crate) fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}
