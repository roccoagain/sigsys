//! A small signals-and-systems toolbox: polynomials, transfer functions,
//! stability, steady-state error, and time/frequency response.
//!
//! ```
//! use controls::Tf;
//!
//! let g = Tf::new([2.0], [1.0, 0.0, 9.0]); // 2 / (s^2 + 9)
//! let d = Tf::new([1.0, 10.0], [1.0]);     // K_d s + K
//! let closed = (&d * &g).unity_feedback();
//! assert!(closed.is_stable());
//! ```

pub mod poly;
mod sim;
pub mod tf;

pub use num_complex::Complex64;
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
