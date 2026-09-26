//! A small signals-and-systems toolbox: polynomials, transfer functions,
//! stability, steady-state error, time and frequency response, root locus,
//! discrete-time systems, signal utilities, and SVG plots.
//!
//! ```
//! use controls::{Input, Tf};
//!
//! let s = Tf::s();
//! let g = 2.0 / (&s * &s + 9.0);   // 2 / (s^2 + 9)
//! let l = Tf::pid(10.0, 0.0, 1.0) * &g; // K = 10, K_d = 1
//! assert!(l.unity_feedback().is_stable());
//! assert!((l.steady_state_error(Input::Step).unwrap() - 9.0 / 29.0).abs() < 1e-12);
//! ```

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
