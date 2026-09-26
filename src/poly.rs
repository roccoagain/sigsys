use std::fmt::{self, Write as _};
use std::ops::{Add, Mul, Neg, Sub};

use num_complex::Complex64;

use crate::forward_owned_op;

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

    pub fn pow(&self, k: usize) -> Poly {
        (0..k).fold(Poly::new([1.0]), |acc, _| &acc * self)
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

    /// All complex roots, via Durand–Kerner iteration. Repeated roots converge
    /// more slowly and are less accurate (around 1e-6).
    pub fn roots(&self) -> Vec<Complex64> {
        let at_origin = self.zeros_at_origin();
        let reduced = &self.coeffs[..self.coeffs.len() - at_origin];
        let n = reduced.len() - 1;
        let mut roots = vec![Complex64::new(0.0, 0.0); at_origin];
        if n == 0 {
            return roots;
        }

        let monic: Vec<f64> = reduced.iter().map(|c| c / reduced[0]).collect();
        let p = Poly { coeffs: monic };
        let seed = Complex64::new(0.4, 0.9);
        let mut r: Vec<Complex64> = (0..n).map(|k| seed.powu(k as u32)).collect();
        for _ in 0..2000 {
            let mut max_step: f64 = 0.0;
            for i in 0..n {
                let denom: Complex64 = (0..n).filter(|j| *j != i).map(|j| r[i] - r[j]).product();
                let step = p.eval(r[i]) / denom;
                r[i] -= step;
                max_step = max_step.max(step.norm());
            }
            if max_step < 1e-14 {
                break;
            }
        }

        for z in &mut r {
            if z.im.abs() < 1e-9 * z.re.abs().max(1.0) {
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
        let p = &self.coeffs;
        let eps = 1e-12 * p.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        let mut rows: Vec<Vec<f64>> = vec![
            p.iter().copied().step_by(2).collect(),
            p.iter().copied().skip(1).step_by(2).collect(),
        ];
        for _ in 2..p.len() {
            let (a, b) = (&rows[rows.len() - 2], &rows[rows.len() - 1]);
            if b[0].abs() <= eps {
                return false;
            }
            let get = |r: &Vec<f64>, j: usize| r.get(j).copied().unwrap_or(0.0);
            let next = (0..a.len())
                .map(|j| (b[0] * get(a, j + 1) - a[0] * get(b, j + 1)) / b[0])
                .collect();
            rows.push(next);
        }
        rows.iter()
            .take(p.len())
            .all(|r| r.first().is_some_and(|c| c.abs() > eps && c.signum() == p[0].signum()))
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
            let mag = c.abs();
            if mag != 1.0 || power == 0 {
                out.push_str(&fmt_coeff(mag));
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

impl Add for &Poly {
    type Output = Poly;
    fn add(self, rhs: &Poly) -> Poly {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let pad = |p: &[f64]| [vec![0.0; n - p.len()], p.to_vec()].concat();
        let sum: Vec<f64> = pad(&self.coeffs).iter().zip(pad(&rhs.coeffs)).map(|(a, b)| a + b).collect();
        Poly::new(sum)
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

forward_owned_op!(Poly, Add, add);
forward_owned_op!(Poly, Mul, mul);
forward_owned_op!(Poly, Sub, sub);

impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.fmt_var("s"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_and_display() {
        let a = Poly::new([1.0, 1.0]); // s + 1
        let b = Poly::new([1.0, 2.0]); // s + 2
        assert_eq!(&a * &b, Poly::new([1.0, 3.0, 2.0]));
        assert_eq!(&a - &b, Poly::new([-1.0]));
        assert_eq!(a.pow(2), Poly::new([1.0, 2.0, 1.0]));
        assert_eq!(Poly::new([0.0, 1.0, 0.0, 9.0]).to_string(), "s^2 + 9");
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
    fn routh() {
        assert!(Poly::new([1.0, 6.0, 11.0, 6.0]).is_hurwitz());
        assert!(!Poly::new([1.0, 1.0, 2.0, 24.0]).is_hurwitz()); // two RHP roots
        assert!(!Poly::new([1.0, 0.0, 9.0]).is_hurwitz()); // poles on jω axis
    }
}
