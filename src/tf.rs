use std::f64::consts::PI;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_complex::Complex64;

use crate::forward_ops;
use crate::poly::Poly;

/// A transfer function `num(s) / den(s)`, stored with a monic denominator.
/// No pole-zero cancellation is performed, and `==` compares coefficients
/// exactly, so `(s+1)/(s+1)^2 != 1/(s+1)`.
///
/// Combine with `*` (series), `+` (parallel), `-`, `/`, and `f64` scalars:
/// `2.0 / (&s * &s + 9.0)` where `s = Tf::s()`.
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
        let lead = den.coeffs()[0];
        Tf { num: num.div_scalar(lead), den: den.div_scalar(lead) }
    }

    pub fn gain(k: f64) -> Self {
        Tf::new([k], [1.0])
    }

    /// The Laplace variable `s`, for building transfer functions algebraically.
    pub fn s() -> Self {
        Tf::new([1.0, 0.0], [1.0])
    }

    /// `k * prod(s - z) / prod(s - p)`. Complex values should come in conjugate pairs.
    pub fn zpk(zeros: &[Complex64], poles: &[Complex64], k: f64) -> Self {
        Tf::from_polys(Poly::from_roots(zeros).scale(k), Poly::from_roots(poles))
    }

    /// PID controller `kp + ki/s + kd s`. Improper when `kd != 0`, so it can be
    /// analyzed and combined but not simulated on its own.
    pub fn pid(kp: f64, ki: f64, kd: f64) -> Self {
        if ki == 0.0 { Tf::new([kd, kp], [1.0]) } else { Tf::new([kd, kp, ki], [1.0, 0.0]) }
    }

    /// `k (s + zero) / (s + pole)`: lead when `zero < pole`, lag when `zero > pole`.
    pub fn lead_lag(k: f64, zero: f64, pole: f64) -> Self {
        Tf::new([k, k * zero], [1.0, pole])
    }

    /// Butterworth low-pass filter of order `n`, cutoff `wc` rad/s, unity DC gain.
    pub fn butterworth(n: usize, wc: f64) -> Self {
        let poles: Vec<Complex64> =
            (0..n).map(|k| Complex64::from_polar(wc, PI / 2.0 + (2 * k + 1) as f64 * PI / (2 * n) as f64)).collect();
        Tf::zpk(&[], &poles, wc.powi(n as i32))
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

    /// Value at `s = 0`: ±infinity if there is a pole at the origin, 0 for a zero there.
    pub fn dc_gain(&self) -> f64 {
        ratio_at_zero(&self.num, &self.den)
    }

    /// Negative feedback: `self / (1 + self * h)`.
    ///
    /// Panics if `1 + self * h` is identically zero (an algebraic loop, such as
    /// unity feedback around a gain of -1).
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
    /// negative-feedback loop. `None` if the closed loop is unstable, which
    /// includes an uncancelled pole-zero pair at the origin such as `s / (s(s+1))`.
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
        let k = if nz > dz { 0.0 } else { self.num.trailing_coeff() / self.den.trailing_coeff() };
        Some(if n == 0 { 1.0 / (1.0 + k) } else { 1.0 / k })
    }

    /// `G(jω)`.
    pub fn freq_response(&self, w: f64) -> Complex64 {
        self.eval(Complex64::new(0.0, w))
    }

    /// Magnitude in dB and phase in degrees (wrapped to (-180, 180]) at `w` rad/s.
    /// See [`Tf::bode_sweep`] for unwrapped phase over a range.
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

impl Neg for &Tf {
    type Output = Tf;
    fn neg(self) -> Tf {
        Tf { num: -&self.num, den: self.den.clone() }
    }
}

impl Neg for Tf {
    type Output = Tf;
    fn neg(self) -> Tf {
        -&self
    }
}

impl Sub for &Tf {
    type Output = Tf;
    fn sub(self, rhs: &Tf) -> Tf {
        self + &(-rhs)
    }
}

impl Div for &Tf {
    type Output = Tf;
    fn div(self, rhs: &Tf) -> Tf {
        Tf::from_polys(&self.num * &rhs.den, &self.den * &rhs.num)
    }
}

forward_ops!(Tf, scalar Tf::gain; Add add, Sub sub, Mul mul, Div div);

/// `lim s->0 num(s)/den(s)`, from the lowest-order nonzero coefficients.
pub(crate) fn ratio_at_zero(num: &Poly, den: &Poly) -> f64 {
    if num.is_zero() {
        return 0.0;
    }
    let ratio = num.trailing_coeff() / den.trailing_coeff();
    match num.zeros_at_origin().cmp(&den.zeros_at_origin()) {
        std::cmp::Ordering::Greater => 0.0,
        std::cmp::Ordering::Equal => ratio,
        std::cmp::Ordering::Less => f64::INFINITY.copysign(ratio),
    }
}

/// Lays out `num` over `den` with a dividing line, both centered.
pub(crate) fn fraction(num: &str, den: &str) -> String {
    let width = num.chars().count().max(den.chars().count());
    format!("{num:^width$}\n{}\n{den:^width$}", "-".repeat(width))
}

impl fmt::Display for Tf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&fraction(&self.num.fmt_var("s"), &self.den.fmt_var("s")))
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
        assert_eq!(Tf::new([3.0], [5.0, 5.0]), Tf::new([0.6], [1.0, 1.0]));
    }

    #[test]
    fn dc_gain_limits() {
        assert_eq!(Tf::new([1.0], [1.0, 0.0]).dc_gain(), f64::INFINITY);
        assert_eq!(Tf::new([-2.0], [1.0, 0.0]).dc_gain(), f64::NEG_INFINITY);
        assert_eq!(Tf::new([1.0, 0.0], [1.0, 1.0]).dc_gain(), 0.0);
        assert_eq!(Tf::new([2.0, 0.0], [1.0, 3.0, 0.0]).dc_gain(), 2.0 / 3.0); // s cancels in the limit
    }

    #[test]
    fn algebraic_construction() {
        let s = Tf::s();
        assert_eq!(2.0 / (&s * &s + 9.0), Tf::new([2.0], [1.0, 0.0, 9.0]));
        assert_eq!(Tf::pid(3.0, 2.0, 1.0), Tf::new([1.0, 3.0, 2.0], [1.0, 0.0]));
        assert_eq!(Tf::pid(3.0, 0.0, 1.0), Tf::new([1.0, 3.0], [1.0]));
        assert_eq!(Tf::lead_lag(2.0, 1.0, 10.0), 2.0 * (&s + 1.0) / (&s + 10.0));
    }

    #[test]
    fn butterworth_is_3db_at_cutoff() {
        for n in 1..=5 {
            let (mag, _) = Tf::butterworth(n, 2.0).bode(2.0);
            assert!((mag + 3.0103).abs() < 1e-3, "order {n}: {mag}");
        }
        assert_eq!(Tf::butterworth(2, 1.0).den.coeffs().len(), 3);
    }

    #[test]
    fn bode_first_order() {
        let (mag, phase) = Tf::new([1.0], [1.0, 1.0]).bode(1.0); // corner frequency
        assert!((mag + 3.0103).abs() < 1e-3);
        assert!((phase + 45.0).abs() < 1e-9);
    }
}
