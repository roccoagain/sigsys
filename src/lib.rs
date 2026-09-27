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
