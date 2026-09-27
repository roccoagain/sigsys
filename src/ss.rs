//! Small dense state-space helpers used internally for simulation and discretization.

use crate::Error;
use crate::tf::Tf;

pub(crate) type Mat = Vec<Vec<f64>>;

/// Single-input single-output state space `x' = Ax + Bu`, `y = Cx + Du`.
pub(crate) struct Ss {
    pub a: Mat,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: f64,
}

impl Tf {
    /// Controllable canonical form, or `Err(Improper)`.
    pub(crate) fn to_ss(&self) -> Result<Ss, Error> {
        let n = self.den.degree();
        if self.num.degree() > n {
            return Err(Error::Improper);
        }

        // Ascending-power coefficients; den is monic.
        let a_asc: Vec<f64> = self.den.coeffs().iter().rev().copied().collect();
        let mut b_asc: Vec<f64> = self.num.coeffs().iter().rev().copied().collect();
        b_asc.resize(n + 1, 0.0);
        let d = b_asc[n];

        let mut a = zeros(n);
        for k in 0..n {
            if k + 1 < n {
                a[k][k + 1] = 1.0;
            }
            a[n - 1][k] = -a_asc[k];
        }
        let mut b = vec![0.0; n];
        if n > 0 {
            b[n - 1] = 1.0;
        }
        let c = (0..n).map(|k| b_asc[k] - d * a_asc[k]).collect();
        Ok(Ss { a, b, c, d })
    }
}

pub(crate) fn zeros(n: usize) -> Mat {
    vec![vec![0.0; n]; n]
}

pub(crate) fn identity(n: usize) -> Mat {
    let mut m = zeros(n);
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

pub(crate) fn matmul(a: &Mat, b: &Mat) -> Mat {
    let n = a.len();
    let mut out = zeros(n);
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

pub(crate) fn mat_vec(a: &Mat, v: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, v)).collect()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Exact discretization over one step `dt` for an input that varies linearly
/// across the step: `x⁺ = Φx + Γ1 u_k + Γ2 (u_{k+1} - u_k)`. With a held input
/// the `Γ2` term vanishes and `(Φ, Γ1)` is the zero-order-hold discretization.
pub(crate) fn discretize(sys: &Ss, dt: f64) -> (Mat, Vec<f64>, Vec<f64>) {
    // exp([[A, B, 0], [0, 0, 1], [0, 0, 0]] dt) = [[Φ, Γ1, Γ2], [0, 1, 1], [0, 0, 1]]
    let n = sys.b.len();
    let mut m = zeros(n + 2);
    for (row, (a_row, b)) in m.iter_mut().zip(sys.a.iter().zip(&sys.b)) {
        for (mij, aij) in row.iter_mut().zip(a_row) {
            *mij = aij * dt;
        }
        row[n] = b * dt;
    }
    m[n][n + 1] = 1.0;
    let e = expm(&m);
    let phi = e[..n].iter().map(|row| row[..n].to_vec()).collect();
    let gamma1 = e[..n].iter().map(|row| row[n]).collect();
    let gamma2 = e[..n].iter().map(|row| row[n + 1]).collect();
    (phi, gamma1, gamma2)
}

/// Matrix exponential by scaling and squaring with a Taylor series.
pub(crate) fn expm(m: &Mat) -> Mat {
    let n = m.len();
    let norm = m.iter().map(|r| r.iter().map(|x| x.abs()).sum::<f64>()).fold(0.0, f64::max);
    let squarings = if norm > 0.5 { (norm / 0.5).log2().ceil() as i32 } else { 0 };
    let factor = 0.5_f64.powi(squarings);
    let scaled: Mat = m.iter().map(|r| r.iter().map(|x| x * factor).collect()).collect();

    let mut term = identity(n);
    let mut sum = identity(n);
    for k in 1..=18 {
        term = matmul(&term, &scaled);
        for row in &mut term {
            for x in row.iter_mut() {
                *x /= k as f64;
            }
        }
        for (s_row, t_row) in sum.iter_mut().zip(&term) {
            for (s, t) in s_row.iter_mut().zip(t_row) {
                *s += t;
            }
        }
    }
    for _ in 0..squarings {
        sum = matmul(&sum, &sum);
    }
    sum
}

/// Faddeev–LeVerrier: characteristic polynomial of `a` (ascending coefficients,
/// monic) and matrices `M_1..M_n` with `adj(λI - A) = Σ M_k λ^(n-k)`.
pub(crate) fn char_poly_adj(a: &Mat) -> (Vec<f64>, Vec<Mat>) {
    let n = a.len();
    let mut c = vec![0.0; n + 1];
    c[n] = 1.0;
    let mut ms = Vec::with_capacity(n);
    let mut prev = zeros(n);
    for k in 1..=n {
        let mut m = matmul(a, &prev);
        for (i, row) in m.iter_mut().enumerate() {
            row[i] += c[n - k + 1];
        }
        let am = matmul(a, &m);
        c[n - k] = -(0..n).map(|i| am[i][i]).sum::<f64>() / k as f64;
        ms.push(m.clone());
        prev = m;
    }
    (c, ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expm_of_rotation() {
        let t = 0.7;
        let e = expm(&vec![vec![0.0, t], vec![-t, 0.0]]);
        assert!((e[0][0] - t.cos()).abs() < 1e-12);
        assert!((e[0][1] - t.sin()).abs() < 1e-12);
    }

    #[test]
    fn char_poly_of_companion() {
        let ss = Tf::new([1.0], [1.0, 6.0, 11.0, 6.0]).to_ss().unwrap();
        let (c, _) = char_poly_adj(&ss.a);
        for (got, want) in c.iter().zip([6.0, 11.0, 6.0, 1.0]) {
            assert!((got - want).abs() < 1e-12);
        }
    }
}
