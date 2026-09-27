#![doc = include_str!("../README.md")]

pub mod discrete;
pub mod freq;
pub mod metrics;
pub mod plot;
pub mod poly;
mod rlocus;
pub mod signals;
mod sim;
mod ss;
pub mod tf;

pub use discrete::{Discretize, Dtf};
pub use freq::{BodePoint, Crossing, Margins};
pub use metrics::StepInfo;
pub use num_complex::Complex64;
pub use plot::{Marker, Plot, save_stacked};
pub use poly::Poly;
pub use tf::{Input, Tf};

/// Why an operation on a transfer function could not be carried out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The numerator has higher degree than the denominator, so there is no
    /// state-space form (continuous) or causal difference equation (discrete)
    /// to simulate or discretize.
    Improper,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Improper => f.write_str("transfer function is improper (numerator degree exceeds denominator)"),
        }
    }
}

impl std::error::Error for Error {}

/// Implements the owned and mixed-reference versions of each operator by
/// forwarding to the `&T op &T` impl. With `scalar $gain`, also implements
/// `T op f64` and `f64 op T` (owned and borrowed), lifting the `f64` with `$gain`.
macro_rules! forward_ops {
    ($t:ty; $($trait:ident $method:ident),*) => {$(
        impl std::ops::$trait for $t {
            type Output = $t;
            fn $method(self, rhs: $t) -> $t { <&$t as std::ops::$trait<&$t>>::$method(&self, &rhs) }
        }
        impl std::ops::$trait<&$t> for $t {
            type Output = $t;
            fn $method(self, rhs: &$t) -> $t { <&$t as std::ops::$trait<&$t>>::$method(&self, rhs) }
        }
        impl std::ops::$trait<$t> for &$t {
            type Output = $t;
            fn $method(self, rhs: $t) -> $t { <&$t as std::ops::$trait<&$t>>::$method(self, &rhs) }
        }
    )*};
    ($t:ty, scalar $gain:path; $($trait:ident $method:ident),*) => {
        forward_ops!($t; $($trait $method),*);
        $(
            impl std::ops::$trait<f64> for &$t {
                type Output = $t;
                fn $method(self, k: f64) -> $t { <&$t as std::ops::$trait<&$t>>::$method(self, &$gain(k)) }
            }
            impl std::ops::$trait<f64> for $t {
                type Output = $t;
                fn $method(self, k: f64) -> $t { <&$t as std::ops::$trait<&$t>>::$method(&self, &$gain(k)) }
            }
            impl std::ops::$trait<&$t> for f64 {
                type Output = $t;
                fn $method(self, rhs: &$t) -> $t { <&$t as std::ops::$trait<&$t>>::$method(&$gain(self), rhs) }
            }
            impl std::ops::$trait<$t> for f64 {
                type Output = $t;
                fn $method(self, rhs: $t) -> $t { <&$t as std::ops::$trait<&$t>>::$method(&$gain(self), &rhs) }
            }
        )*
    };
}
pub(crate) use forward_ops;

/// Deterministic pseudo-random numbers for property-style tests (xorshift64).
#[cfg(test)]
pub(crate) struct TestRng(u64);

#[cfg(test)]
impl TestRng {
    pub(crate) fn new(seed: u64) -> Self {
        TestRng(seed.max(1))
    }

    /// Uniform in `[lo, hi)`.
    pub(crate) fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        lo + (hi - lo) * (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `lo..=hi`.
    pub(crate) fn int(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.uniform(0.0, (hi - lo + 1) as f64) as usize).min(hi - lo)
    }
}
