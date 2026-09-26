use std::fmt;
use std::ops::{Add, Mul};

use num_complex::Complex64;

use crate::forward_owned_op;
use crate::poly::Poly;

/// A transfer function `num(s) / den(s)`, stored with a monic denominator.
/// No pole-zero cancellation is performed.
#[derive(Clone, Debug, PartialEq)]
pub struct Tf {
    pub num: Poly,
    pub den: Poly,
}

/// Test input for steady-state error: `1/s`, `1/s^2`, `1/s^3`.
#[derive(Clone, Copy, Debug)]
pub enum Input {
    Step = 0,
    Ramp = 1,
    Parabola = 2,
}

impl Tf {
    /// Coefficients highest power first. Panics if the denominator is zero.
    pub fn new(num: impl Into<Vec<f64>>, den: impl Into<Vec<f64>>) -> Self {
        Tf::from_polys(Poly::new(num), Poly::new(den))
    }

    pub fn from_polys(num: Poly, den: Poly) -> Self {
        assert!(!den.is_zero(), "transfer function denominator is zero");
        let lead = Poly::new([1.0 / den.coeffs()[0]]);
        Tf { num: &num * &lead, den: &den * &lead }
    }

    pub fn gain(k: f64) -> Self {
        Tf::new([k], [1.0])
    }

    pub fn eval(&self, s: Complex64) -> Complex64 {
        self.num.eval(s) / self.den.eval(s)
    }

    pub fn poles(&self) -> Vec<Complex64> {
        self.den.roots()
    }

    pub fn zeros(&self) -> Vec<Complex64> {
        self.num.roots()
    }

    /// Value at `s = 0` (infinite if there is a pole at the origin).
    pub fn dc_gain(&self) -> f64 {
        self.eval(Complex64::new(0.0, 0.0)).re
    }

    /// Negative feedback: `self / (1 + self * h)`.
    pub fn feedback(&self, h: &Tf) -> Tf {
        let num = &self.num * &h.den;
        let den = &(&self.den * &h.den) + &(&self.num * &h.num);
        Tf::from_polys(num, den)
    }

    pub fn unity_feedback(&self) -> Tf {
        self.feedback(&Tf::gain(1.0))
    }

    /// True iff every pole is in the open left half-plane.
    pub fn is_stable(&self) -> bool {
        self.den.is_hurwitz()
    }

    /// Number of poles at the origin (net of zeros there).
    pub fn system_type(&self) -> usize {
        self.den.zeros_at_origin().saturating_sub(self.num.zeros_at_origin())
    }

    /// Steady-state error when `self` is the open loop `L(s)` in a unity
    /// negative-feedback loop. `None` if the closed loop is unstable.
    pub fn steady_state_error(&self, input: Input) -> Option<f64> {
        if !self.unity_feedback().is_stable() {
            return None;
        }
        let (nz, dz) = (self.num.zeros_at_origin(), self.den.zeros_at_origin());
        let n = input as usize;
        let sys_type = self.system_type();
        if sys_type > n {
            return Some(0.0);
        }
        if sys_type < n {
            return Some(f64::INFINITY);
        }
        // Error constant: lim s->0 of s^n L(s), from the lowest nonzero coefficients.
        let k = if nz > dz {
            0.0
        } else {
            let (num, den) = (self.num.coeffs(), self.den.coeffs());
            num[num.len() - 1 - nz] / den[den.len() - 1 - dz]
        };
        Some(if n == 0 { 1.0 / (1.0 + k) } else { 1.0 / k })
    }

    /// `G(jω)`.
    pub fn freq_response(&self, w: f64) -> Complex64 {
        self.eval(Complex64::new(0.0, w))
    }

    /// Magnitude in dB and phase in degrees (wrapped to (-180, 180]) at `w` rad/s.
    pub fn bode(&self, w: f64) -> (f64, f64) {
        let g = self.freq_response(w);
        (20.0 * g.norm().log10(), g.arg().to_degrees())
    }
}

impl Mul for &Tf {
    type Output = Tf;
    /// Series connection.
    fn mul(self, rhs: &Tf) -> Tf {
        Tf::from_polys(&self.num * &rhs.num, &self.den * &rhs.den)
    }
}

impl Add for &Tf {
    type Output = Tf;
    /// Parallel connection.
    fn add(self, rhs: &Tf) -> Tf {
        let num = &(&self.num * &rhs.den) + &(&rhs.num * &self.den);
        Tf::from_polys(num, &self.den * &rhs.den)
    }
}

forward_owned_op!(Tf, Mul, mul);
forward_owned_op!(Tf, Add, add);

impl fmt::Display for Tf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (num, den) = (self.num.to_string(), self.den.to_string());
        let width = num.len().max(den.len());
        writeln!(f, "{num:^width$}")?;
        writeln!(f, "{}", "-".repeat(width))?;
        write!(f, "{den:^width$}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pd_on_undamped_plant() {
        let g = Tf::new([2.0], [1.0, 0.0, 9.0]);
        let l = &Tf::new([1.0, 10.0], [1.0]) * &g; // K = 10, K_d = 1
        let e = l.steady_state_error(Input::Step).unwrap();
        assert!((e - 9.0 / 29.0).abs() < 1e-12);
        assert_eq!(l.steady_state_error(Input::Ramp), Some(f64::INFINITY));
        let no_kd = &Tf::gain(10.0) * &g;
        assert_eq!(no_kd.steady_state_error(Input::Step), None);
    }

    #[test]
    fn type_one_system() {
        // D = (2s + 1)/s on G = 1/(s + 1): type 1, K_v = 1
        let l = Tf::new([2.0, 1.0], [1.0, 0.0]) * Tf::new([1.0], [1.0, 1.0]);
        assert_eq!(l.system_type(), 1);
        assert_eq!(l.steady_state_error(Input::Step), Some(0.0));
        assert_eq!(l.steady_state_error(Input::Ramp), Some(1.0));
        assert_eq!(l.steady_state_error(Input::Parabola), Some(f64::INFINITY));
    }

    #[test]
    fn interconnections() {
        let g = Tf::new([1.0], [1.0, 1.0]);
        assert_eq!(g.unity_feedback(), Tf::new([1.0], [1.0, 2.0]));
        assert_eq!(&g + &g, Tf::new([2.0, 2.0], [1.0, 2.0, 1.0]));
        assert_eq!(Tf::new([4.0], [2.0, 2.0]), Tf::new([2.0], [1.0, 1.0])); // normalized
        assert!((g.dc_gain() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bode_first_order() {
        let (mag, phase) = Tf::new([1.0], [1.0, 1.0]).bode(1.0); // corner frequency
        assert!((mag + 3.0103).abs() < 1e-3);
        assert!((phase + 45.0).abs() < 1e-9);
    }
}
