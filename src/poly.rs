use std::f64::consts::PI;
use std::fmt::{self, Write as _};
use std::ops::{Add, Mul, Neg, Sub};

use num_complex::Complex64;

use crate::forward_ops;

/// A polynomial, coefficients highest power first: `s^2 + 9` is `[1, 0, 9]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    coeffs: Vec<f64>,
}

impl Poly {
    /// Leading zeros are trimmed; an empty or all-zero list is the zero polynomial.
    pub fn new(coeffs: impl Into<Vec<f64>>) -> Self {
        let c = coeffs.into();
        let coeffs = match c.iter().position(|x| *x != 0.0) {
            Some(i) => c[i..].to_vec(),
            None => vec![0.0],
        };
        Poly { coeffs }
    }

    /// Monic polynomial with the given roots. Complex roots should come in
    /// conjugate pairs; any leftover imaginary part is dropped.
    pub fn from_roots(roots: &[Complex64]) -> Self {
        let mut c = vec![Complex64::new(1.0, 0.0)];
        for r in roots {
            let mut next = vec![Complex64::new(0.0, 0.0); c.len() + 1];
            for (i, ci) in c.iter().enumerate() {
                next[i] += ci;
                next[i + 1] -= ci * r;
            }
            c = next;
        }
        Poly::new(c.iter().map(|z| z.re).collect::<Vec<_>>())
    }

    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs == [0.0]
    }

    pub fn scale(&self, k: f64) -> Poly {
        Poly::new(self.coeffs.iter().map(|c| c * k).collect::<Vec<_>>())
    }

    /// Divides every coefficient by `k` (dividing, not multiplying by `1/k`, keeps
    /// results exact when they are representable, e.g. 3/5 == 0.6).
    pub(crate) fn div_scalar(&self, k: f64) -> Poly {
        Poly::new(self.coeffs.iter().map(|c| c / k).collect::<Vec<_>>())
    }

    pub fn pow(&self, k: usize) -> Poly {
        (0..k).fold(Poly::new([1.0]), |acc, _| &acc * self)
    }

    pub fn derivative(&self) -> Poly {
        let n = self.degree();
        Poly::new(self.coeffs[..n].iter().enumerate().map(|(i, c)| c * (n - i) as f64).collect::<Vec<_>>())
    }

    /// `p(s + a)` (Taylor shift), e.g. to examine behaviour near `s = -a`.
    pub fn shift(&self, a: f64) -> Poly {
        let mut c = self.coeffs.clone();
        let n = c.len();
        for i in 0..n {
            for j in 1..n - i {
                c[j] += a * c[j - 1];
            }
        }
        Poly::new(c)
    }

    /// `p(a/b) b^n` for polynomials `a`, `b`: substitutes the rational function
    /// `a/b` for the variable and clears denominators. `n` must be at least the
    /// degree of `p` (pass a larger `n` to put two polynomials over a common `b^n`).
    pub(crate) fn compose_ratio(&self, a: &Poly, b: &Poly, n: usize) -> Poly {
        let d = self.degree();
        self.coeffs.iter().enumerate().fold(Poly::new([0.0]), |acc, (i, &c)| {
            let k = d - i;
            acc + (&a.pow(k) * &b.pow(n - k)).scale(c)
        })
    }

    pub fn eval(&self, s: Complex64) -> Complex64 {
        self.coeffs.iter().fold(Complex64::new(0.0, 0.0), |acc, c| acc * s + c)
    }

    /// Number of roots at the origin.
    pub fn zeros_at_origin(&self) -> usize {
        if self.is_zero() {
            return 0;
        }
        self.coeffs.iter().rev().take_while(|c| **c == 0.0).count()
    }

    /// Lowest-order nonzero coefficient, which sets the behaviour near `s = 0`
    /// (0 for the zero polynomial).
    pub fn trailing_coeff(&self) -> f64 {
        self.coeffs[self.coeffs.len() - 1 - self.zeros_at_origin()]
    }

    /// All complex roots, via Durand–Kerner iteration. Simple roots are accurate
    /// to near machine precision (relative to the polynomial's conditioning). A root of multiplicity `m` is inherently
    /// ill-conditioned: expect errors around `1e-16^(1/m)` (about 1e-4 for a
    /// 4-fold root), possibly as a small spurious imaginary part.
    pub fn roots(&self) -> Vec<Complex64> {
        let at_origin = self.zeros_at_origin();
        let reduced = &self.coeffs[..self.coeffs.len() - at_origin];
        let n = reduced.len() - 1;
        let mut roots = vec![Complex64::new(0.0, 0.0); at_origin];
        if n == 0 {
            return roots;
        }

        let monic: Vec<f64> = reduced.iter().map(|c| c / reduced[0]).collect();
        // Substitute s = σt with σ the geometric mean of the root magnitudes, so the
        // roots in t are of order 1 whether they sit near 1e-6 or 1e6. Started near
        // the unit circle, the iteration then can't overshoot far enough to overflow.
        let sigma = monic[n].abs().powf(1.0 / n as f64);
        let scaled: Vec<f64> = monic.iter().enumerate().map(|(i, c)| c / sigma.powi(i as i32)).collect();
        let (p, sigma) = if scaled.iter().all(|c| c.is_finite()) {
            (Poly { coeffs: scaled }, sigma)
        } else {
            (Poly { coeffs: monic }, 1.0)
        };
        // If one starting circle fails, retry from another rather than return NaN.
        let mut r = [(1.0, 0.4), (0.5, 1.1), (2.0, 2.3)]
            .into_iter()
            .map(|(radius, angle)| durand_kerner(&p, radius, angle))
            .find(|r| r.iter().all(|z| z.is_finite()))
            .expect("root finding diverged from every starting point");
        for z in &mut r {
            *z *= sigma;
        }

        for z in &mut r {
            if z.im.abs() < 1e-9 * z.norm() {
                z.im = 0.0;
            }
        }
        roots.extend(r);
        roots.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
        roots
    }

    /// Routh–Hurwitz: true iff every root is in the open left half-plane, i.e.
    /// every first-column entry of the Routh array is nonzero with the same sign.
    pub fn is_hurwitz(&self) -> bool {
        let n = self.degree();
        let c = &self.coeffs;
        if self.is_zero() || c[n] == 0.0 {
            return false; // root at the origin
        }
        // Substitute s = σs' with σ the geometric mean of the root magnitudes, so
        // the array is well scaled whether the roots are near 1e-6 or 1e6.
        let sigma = (c[n] / c[0]).abs().powf(1.0 / n.max(1) as f64);
        let scaled: Vec<f64> = c.iter().enumerate().map(|(i, x)| x * sigma.powi((n - i) as i32)).collect();
        let max = scaled.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let p: Vec<f64> = scaled.iter().map(|x| x / max).collect();
        // Each entry carries a bound on its accumulated rounding error, so an entry
        // counts as zero only when it is indistinguishable from cancellation noise.
        // Given coefficients are exact: only scaling has perturbed them.
        let noise: Vec<f64> = p.iter().map(|x| x.abs() * (n + 2) as f64 * f64::EPSILON).collect();
        let is_zero = |x: f64, e: f64| x == 0.0 || x.abs() <= 16.0 * e;
        let split = |v: &[f64], skip: usize| -> Vec<f64> { v.iter().copied().skip(skip).step_by(2).collect() };
        let mut rows = vec![split(&p, 0), split(&p, 1)];
        let mut errs = vec![split(&noise, 0), split(&noise, 1)];
        let get = |r: &Vec<f64>, j: usize| r.get(j).copied().unwrap_or(0.0);
        for _ in 2..p.len() {
            let (a, b) = (&rows[rows.len() - 2], &rows[rows.len() - 1]);
            let (ea, eb) = (&errs[errs.len() - 2], &errs[errs.len() - 1]);
            if is_zero(b[0], eb[0]) {
                return false;
            }
            let (next, next_err) = (0..a.len())
                .map(|j| {
                    let (a1, b1) = (get(a, j + 1), get(b, j + 1));
                    let x = (b[0] * a1 - a[0] * b1) / b[0];
                    let rounding = f64::EPSILON * (b[0] * a1).abs().max((a[0] * b1).abs());
                    let carried = b[0].abs() * get(ea, j + 1)
                        + a1.abs() * eb[0]
                        + a[0].abs() * get(eb, j + 1)
                        + b1.abs() * ea[0]
                        + x.abs() * eb[0];
                    (x, (rounding + carried) / b[0].abs())
                })
                .unzip();
            rows.push(next);
            errs.push(next_err);
        }
        rows.iter()
            .zip(&errs)
            .take(p.len())
            .all(|(r, e)| r.first().is_some_and(|c| !is_zero(*c, e[0]) && c.signum() == p[0].signum()))
    }

    /// Formats with the given variable name, e.g. `fmt_var("z")` gives `z^2 + 9`.
    pub fn fmt_var(&self, var: &str) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let n = self.degree();
        let mut out = String::new();
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let power = n - i;
            match (out.is_empty(), c < 0.0) {
                (true, true) => out.push('-'),
                (true, false) => {}
                (false, neg) => out.push_str(if neg { " - " } else { " + " }),
            }
            let mag = fmt_coeff(c.abs());
            if mag != "1" || power == 0 {
                out.push_str(&mag);
            }
            match power {
                0 => {}
                1 => out.push_str(var),
                _ => write!(out, "{var}^{power}").unwrap(),
            }
        }
        out
    }
}

/// Durand–Kerner iteration for the roots of monic `p`, from `degree` starting
/// points evenly spaced on a circle of the given radius and rotated by `angle`
/// (off the real axis, so they aren't conjugate-symmetric). Stops early if an
/// iterate stops being finite.
fn durand_kerner(p: &Poly, radius: f64, angle: f64) -> Vec<Complex64> {
    let n = p.degree();
    let mut r: Vec<Complex64> =
        (0..n).map(|k| Complex64::from_polar(radius, angle + 2.0 * PI * k as f64 / n as f64)).collect();
    for _ in 0..2000 {
        let mut max_step: f64 = 0.0;
        for i in 0..n {
            let denom: Complex64 = (0..n).filter(|j| *j != i).map(|j| r[i] - r[j]).product();
            let step = p.eval(r[i]) / denom;
            r[i] -= step;
            max_step = max_step.max(step.norm());
        }
        if max_step < 1e-14 || !max_step.is_finite() || r.iter().any(|z| !z.is_finite()) {
            break;
        }
    }
    r
}

/// Five significant figures, trailing zeros trimmed; scientific notation when tiny or huge.
fn fmt_coeff(x: f64) -> String {
    let exp = x.abs().log10().floor();
    if !(-4.0..9.0).contains(&exp) {
        return format!("{x:.4e}");
    }
    let decimals = (4.0 - exp).max(0.0) as usize;
    let s = format!("{x:.decimals$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// Coefficient-wise sum in which results that cancel to rounding noise (relative
/// to the two terms being added) become exact zeros, so s(s + 0.3) built by
/// subtraction still has a root at 0.
fn sum_coeffs(a: &Poly, b: &Poly) -> Vec<f64> {
    let n = a.coeffs.len().max(b.coeffs.len());
    let pad = |p: &[f64]| [vec![0.0; n - p.len()], p.to_vec()].concat();
    pad(&a.coeffs)
        .iter()
        .zip(pad(&b.coeffs))
        .map(|(x, y)| if (x + y).abs() <= 1e-14 * x.abs().max(y.abs()) { 0.0 } else { x + y })
        .collect()
}

impl Add for &Poly {
    type Output = Poly;
    fn add(self, rhs: &Poly) -> Poly {
        Poly::new(sum_coeffs(self, rhs))
    }
}

impl Mul for &Poly {
    type Output = Poly;
    fn mul(self, rhs: &Poly) -> Poly {
        let mut out = vec![0.0; self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in rhs.coeffs.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        Poly::new(out)
    }
}

impl Neg for &Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        self.scale(-1.0)
    }
}

impl Sub for &Poly {
    type Output = Poly;
    fn sub(self, rhs: &Poly) -> Poly {
        self + &(-rhs)
    }
}

impl Neg for Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        -&self
    }
}

forward_ops!(Poly; Add add, Sub sub, Mul mul);

impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.fmt_var("s"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestRng;

    /// Largest relative distance from each wanted root to its nearest unused computed root.
    fn worst_match(got: &[Complex64], want: &[Complex64]) -> f64 {
        assert_eq!(got.len(), want.len());
        let mut used = vec![false; got.len()];
        want.iter()
            .map(|w| {
                let (j, e) = (0..got.len())
                    .filter(|j| !used[*j])
                    .map(|j| (j, (got[j] - w).norm() / w.norm().max(1e-300)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                used[j] = true;
                e
            })
            .fold(0.0, f64::max)
    }

    /// `|p(z)|` relative to the size of the terms summed to evaluate it: about
    /// machine epsilon when `z` is an exact root of a slightly perturbed `p`.
    fn backward_error(p: &Poly, z: Complex64) -> f64 {
        let scale = p.coeffs().iter().fold(0.0, |acc, c| acc * z.norm() + c.abs());
        p.eval(z).norm() / scale
    }

    /// Random real roots and conjugate pairs, magnitudes spread over `decades` either side of 1.
    fn random_roots(rng: &mut TestRng, n: usize, decades: f64) -> Vec<Complex64> {
        let mut r = Vec::with_capacity(n);
        while r.len() < n {
            let mag = 10f64.powf(rng.uniform(-decades, decades));
            if r.len() + 2 <= n && rng.uniform(0.0, 1.0) < 0.5 {
                let z = Complex64::from_polar(mag, rng.uniform(0.05, 3.09));
                r.extend([z, z.conj()]);
            } else {
                r.push(Complex64::new(if rng.uniform(0.0, 1.0) < 0.5 { -mag } else { mag }, 0.0));
            }
        }
        r
    }

    #[test]
    fn consecutive_integer_roots_up_to_order_20() {
        // Roots -1..-n: once overflowed to NaN from n = 14. Wilkinson's polynomial
        // is famously ill-conditioned, so forward error grows with n; backward
        // error (how exactly each root satisfies p) should stay near epsilon.
        for n in 1..=20 {
            let want: Vec<Complex64> = (1..=n).map(|k| Complex64::new(-(k as f64), 0.0)).collect();
            let p = Poly::from_roots(&want);
            let got = p.roots();
            assert!(got.iter().all(|z| z.is_finite()), "n = {n}: {got:?}");
            for z in &got {
                assert!(backward_error(&p, *z) < 1e-15, "n = {n}: {z}");
            }
            let tol = if n <= 12 { 1e-8 } else { 1e-2 };
            assert!(worst_match(&got, &want) < tol, "n = {n}");
        }
    }

    #[test]
    fn random_roots_round_trip() {
        let mut rng = TestRng::new(7);
        for decades in [0.5, 1.5, 3.0] {
            for _ in 0..500 {
                let n = rng.int(1, 10);
                let want = random_roots(&mut rng, n, decades);
                let got = Poly::from_roots(&want).roots();
                assert!(worst_match(&got, &want) < 1e-9, "{want:?}\n{got:?}");
            }
        }
    }

    #[test]
    fn routh_agrees_with_roots() {
        let mut rng = TestRng::new(11);
        let mut checked = 0;
        for _ in 0..5000 {
            let n = rng.int(1, 8);
            let c: Vec<f64> = (0..=n).map(|i| if i == 0 { 1.0 } else { rng.uniform(-5.0, 5.0) }).collect();
            let p = Poly::new(c);
            let max_re = p.roots().iter().map(|z| z.re).fold(f64::NEG_INFINITY, f64::max);
            if max_re.abs() < 1e-6 {
                continue; // too close to the boundary for the roots to decide
            }
            checked += 1;
            assert_eq!(p.is_hurwitz(), max_re < 0.0, "{p}");
        }
        assert!(checked > 4900);
    }

    #[test]
    fn sum_keeps_small_terms_that_did_not_cancel() {
        assert_eq!(&Poly::new([1.0, 1e-15]) + &Poly::new([0.0]), Poly::new([1.0, 1e-15]));
        let noisy = &(&Poly::new([1.0, 0.1]) * &Poly::new([1.0, 0.2])) - &Poly::new([0.02]);
        assert_eq!(noisy, Poly::new([1.0, 0.30000000000000004, 0.0]));
    }

    #[test]
    fn small_complex_roots_stay_complex() {
        let r = Poly::new([1.0, 2e-11, 1.01e-20]).roots(); // -1e-11 ± 1e-10 i
        for z in &r {
            assert!((z.re + 1e-11).abs() < 1e-18 && (z.im.abs() - 1e-10).abs() < 1e-18, "{r:?}");
        }
    }

    #[test]
    fn hurwitz_accepts_light_damping_and_rejects_marginal() {
        assert!(Poly::new([1.0, 1e-13, 1.0]).is_hurwitz());
        assert!(!Poly::new([1.0, 0.0, 1.0]).is_hurwitz());
        assert!(!Poly::new([1.0, 1.0, 1.0, 1.0]).is_hurwitz()); // (s + 1)(s^2 + 1)
        assert!(!(&Poly::new([1.0, 0.3]) * &Poly::new([1.0, 0.0, 0.7])).is_hurwitz());
        assert!(!Poly::new([1.0, 1.0, -1.0]).is_hurwitz());
    }

    #[test]
    fn display_never_prints_a_unit_coefficient() {
        assert_eq!(Poly::new([0.999999, 1.0]).to_string(), "s + 1");
    }

    #[test]
    fn arithmetic_and_display() {
        let a = Poly::new([1.0, 1.0]); // s + 1
        let b = Poly::new([1.0, 2.0]); // s + 2
        assert_eq!(&a * &b, Poly::new([1.0, 3.0, 2.0]));
        assert_eq!(&a - &b, Poly::new([-1.0]));
        assert_eq!(a.pow(2), Poly::new([1.0, 2.0, 1.0]));
        assert_eq!(a.clone() * &b, &a * b.clone());
        assert_eq!(-a.clone(), Poly::new([-1.0, -1.0]));
        assert_eq!(Poly::new([3.0, 2.0, 0.0]).trailing_coeff(), 2.0);
        assert_eq!(Poly::new([0.0]).trailing_coeff(), 0.0);
        assert_eq!(Poly::new([0.0, 1.0, 0.0, 9.0]).to_string(), "s^2 + 9");
        assert_eq!(Poly::new([1.0, 3.0, 2.0]).derivative(), Poly::new([2.0, 3.0]));
        assert_eq!(Poly::new([5.0]).derivative(), Poly::new([0.0]));
        assert_eq!(Poly::new([-2.0, 1.0, -0.5]).fmt_var("z"), "-2z^2 + z - 0.5");
    }

    #[test]
    fn roots() {
        let r = Poly::new([1.0, 6.0, 11.0, 6.0]).roots(); // (s+1)(s+2)(s+3)
        for (got, want) in r.iter().zip([-3.0, -2.0, -1.0]) {
            assert!((got.re - want).abs() < 1e-9);
        }
        let r = Poly::new([1.0, 0.0, 9.0, 0.0]).roots(); // s(s^2 + 9)
        assert_eq!(r.len(), 3);
        assert!(r.iter().any(|z| z.norm() == 0.0));
        assert!(r.iter().any(|z| (z.im - 3.0).abs() < 1e-9));
    }

    #[test]
    fn from_roots_round_trips() {
        let roots = [Complex64::new(-1.0, 2.0), Complex64::new(-1.0, -2.0), Complex64::new(-3.0, 0.0)];
        let p = Poly::from_roots(&roots);
        assert_eq!(p, Poly::new([1.0, 5.0, 11.0, 15.0])); // (s^2 + 2s + 5)(s + 3)
    }

    #[test]
    fn shift_and_cancellation() {
        assert_eq!(Poly::new([1.0, 0.0, 0.0]).shift(1.0), Poly::new([1.0, 2.0, 1.0])); // (s+1)^2
        let p = &(&Poly::new([1.0, 0.1]) * &Poly::new([1.0, 0.2])) - &Poly::new([0.02]);
        assert_eq!(p.zeros_at_origin(), 1);
        let q = &Poly::new([0.1 + 0.2, 1.0]) - &Poly::new([0.3, 0.0]); // leading term cancels
        assert_eq!(q.degree(), 0);
    }

    #[test]
    fn routh_is_scale_free() {
        let fast = Poly::new([1.0, 1e3]).pow(5);
        let slow = Poly::new([1.0, 1e-4]).pow(4);
        assert!(fast.is_hurwitz() && slow.is_hurwitz());
        assert!(Poly::new([1.0, 1e-13]).is_hurwitz());
        assert!(!Poly::new([1.0, 1e3, 0.0]).is_hurwitz()); // root at origin
        assert!(Poly::new([-1.0, -3.0, -2.0]).is_hurwitz());
        assert!(!Poly::new([1.0, 1.0, 2.0, 2.0, 3.0]).is_hurwitz()); // zero in first column
    }

    #[test]
    fn routh() {
        assert!(Poly::new([1.0, 6.0, 11.0, 6.0]).is_hurwitz());
        assert!(!Poly::new([1.0, 1.0, 2.0, 24.0]).is_hurwitz()); // two RHP roots
        assert!(!Poly::new([1.0, 0.0, 9.0]).is_hurwitz()); // poles on jω axis
    }
}
