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

/// Implements an owned `a op b` by forwarding to the `&a op &b` impl.
macro_rules! forward_owned_op {
    ($t:ty, $trait:ident, $method:ident) => {
        impl std::ops::$trait for $t {
            type Output = $t;
            fn $method(self, rhs: $t) -> $t {
                (&self).$method(&rhs)
            }
        }
    };
}
pub(crate) use forward_owned_op;
