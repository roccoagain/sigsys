use std::fmt;

use num_complex::Complex64;

use crate::poly::Poly;
use crate::ss;
use crate::tf::{Tf, fraction, ratio_at_zero};

/// Continuous-to-discrete conversion method.
#[derive(Clone, Copy, Debug)]
pub enum Discretize {
    /// Zero-order hold: exact for piecewise-constant inputs.
    Zoh,
    /// Bilinear transform `s = (2/T)(z - 1)/(z + 1)`.
    Tustin,
}

/// A discrete-time transfer function `num(z) / den(z)` with sample time `ts`,
/// stored with a monic denominator.
#[derive(Clone, Debug, PartialEq)]
pub struct Dtf {
    pub num: Poly,
    pub den: Poly,
    pub ts: f64,
}

impl Dtf {
    /// Coefficients highest power of `z` first.
    pub fn new(num: impl Into<Vec<f64>>, den: impl Into<Vec<f64>>, ts: f64) -> Self {
        Dtf::from_polys(Poly::new(num), Poly::new(den), ts)
    }

    pub fn from_polys(num: Poly, den: Poly, ts: f64) -> Self {
        assert!(!den.is_zero(), "transfer function denominator is zero");
        let lead = den.coeffs()[0];
        Dtf { num: num.div_scalar(lead), den: den.div_scalar(lead), ts }
    }

    pub fn eval(&self, z: Complex64) -> Complex64 {
        self.num.eval(z) / self.den.eval(z)
    }

    pub fn poles(&self) -> Vec<Complex64> {
        self.den.roots()
    }

    pub fn zeros(&self) -> Vec<Complex64> {
        self.num.roots()
    }

    /// Value at `z = 1`: ±infinity if there is a pole there (an integrator).
    pub fn dc_gain(&self) -> f64 {
        ratio_at_zero(&self.num.shift(1.0), &self.den.shift(1.0))
    }

    /// True iff every pole is strictly inside the unit circle.
    pub fn is_stable(&self) -> bool {
        self.poles().iter().all(|p| p.norm() < 1.0 - 1e-9)
    }

    /// `H(e^{jωT})` for `w` in rad/s.
    pub fn freq_response(&self, w: f64) -> Complex64 {
        self.eval(Complex64::from_polar(1.0, w * self.ts))
    }

    /// Output for the input sequence `u`, from rest (difference equation).
    /// Panics if the transfer function is non-causal.
    pub fn simulate(&self, u: &[f64]) -> Vec<f64> {
        let n = self.den.degree();
        assert!(self.num.degree() <= n, "non-causal transfer function");
        let a = self.den.coeffs();
        let b = [vec![0.0; n - self.num.degree()], self.num.coeffs().to_vec()].concat();
        let mut y = Vec::with_capacity(u.len());
        for k in 0..u.len() {
            let forced: f64 = (0..=n.min(k)).map(|i| b[i] * u[k - i]).sum();
            let past: f64 = (1..=n.min(k)).map(|i| a[i] * y[k - i]).sum();
            y.push(forced - past);
        }
        y
    }

    /// Unit step response over `samples` samples, as `(t, y)`.
    pub fn step_response(&self, samples: usize) -> (Vec<f64>, Vec<f64>) {
        let t = (0..samples).map(|k| k as f64 * self.ts).collect();
        (t, self.simulate(&vec![1.0; samples]))
    }
}

impl fmt::Display for Dtf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}\n(Ts = {})", fraction(&self.num.fmt_var("z"), &self.den.fmt_var("z")), self.ts)
    }
}

impl Tf {
    /// Discretizes with sample time `ts`. ZOH panics for an improper transfer
    /// function (it has no state-space form); Tustin handles any.
    pub fn c2d(&self, ts: f64, method: Discretize) -> Dtf {
        match method {
            Discretize::Zoh => zoh(self, ts),
            Discretize::Tustin => tustin(self, ts),
        }
    }
}

fn zoh(g: &Tf, ts: f64) -> Dtf {
    let sys = g.to_ss();
    let n = sys.b.len();
    if n == 0 {
        return Dtf::new([sys.d], [1.0], ts);
    }

    let (ad, bd, _) = ss::discretize(&sys, ts);

    // H(z) = C adj(zI - Ad) Bd / det(zI - Ad) + D
    let (char_poly, adj) = ss::char_poly_adj(&ad);
    let mut num: Vec<f64> = char_poly.iter().map(|c| c * sys.d).collect();
    for (k, mk) in adj.iter().enumerate() {
        num[n - 1 - k] += ss::dot(&sys.c, &ss::mat_vec(mk, &bd)); // M_(k+1) multiplies z^(n-1-k)
    }
    let rev = |v: Vec<f64>| v.into_iter().rev().collect::<Vec<_>>();
    Dtf::new(rev(num), rev(char_poly), ts)
}

fn tustin(g: &Tf, ts: f64) -> Dtf {
    let n = g.den.degree().max(g.num.degree());
    let (zm1, zp1) = (Poly::new([1.0, -1.0]), Poly::new([1.0, 1.0]));
    // Σ c_k s^k  ->  Σ c_k (2/T)^k (z-1)^k (z+1)^(n-k), after multiplying through by (z+1)^n.
    let substitute = |p: &Poly| {
        let d = p.degree();
        p.coeffs().iter().enumerate().fold(Poly::new([0.0]), |acc, (i, &c)| {
            let k = d - i;
            acc + (&zm1.pow(k) * &zp1.pow(n - k)).scale(c * (2.0 / ts).powi(k as i32))
        })
    };
    Dtf::from_polys(substitute(&g.num), substitute(&g.den), ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn zoh_first_order_is_exact() {
        let t = 0.1;
        let d = Tf::new([1.0], [1.0, 1.0]).c2d(t, Discretize::Zoh);
        let p = (-t).exp();
        assert!(close(d.num.coeffs()[0], 1.0 - p));
        assert!(close(d.den.coeffs()[1], -p));
    }

    #[test]
    fn zoh_matches_continuous_step_at_samples() {
        let g = Tf::new([1.0, 5.0], [1.0, 2.0, 5.0]);
        let d = g.c2d(0.05, Discretize::Zoh);
        let (_, yd) = d.step_response(101);
        let (_, yc) = g.step_response(5.0, 0.0005);
        for k in 0..=100 {
            assert!((yd[k] - yc[k * 100]).abs() < 1e-8, "sample {k}");
        }
        for p in d.poles() {
            assert!(close(p.norm(), (-0.05_f64).exp())); // |e^{pT}| with Re p = -1
        }
    }

    #[test]
    fn tustin_first_order() {
        let t = 0.1;
        let d = Tf::new([1.0], [1.0, 1.0]).c2d(t, Discretize::Tustin);
        assert!(close(d.dc_gain(), 1.0));
        assert!(close(d.poles()[0].re, (1.0 - t / 2.0) / (1.0 + t / 2.0)));
        assert!(d.is_stable());
    }

    #[test]
    fn integrator_dc_gain_is_infinite() {
        assert_eq!(Dtf::new([1.0], [1.0, -1.0], 0.1).dc_gain(), f64::INFINITY);
    }

    #[test]
    fn difference_equation() {
        // y[k] = 0.5 y[k-1] + u[k-1]
        let y = Dtf::new([1.0], [1.0, -0.5], 1.0).simulate(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(y, vec![0.0, 1.0, 0.5, 0.25]);
    }
}
