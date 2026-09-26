use num_complex::Complex64;

use crate::tf::Tf;

impl Tf {
    /// Closed-loop poles of `1 + k L(s) = 0` for each gain, where `self` is `L(s)`.
    /// Poles are grouped into continuous branches: `branches[b][i]` is branch `b`
    /// at `gains[i]` (matched greedily to the nearest pole at the previous gain).
    pub fn root_locus(&self, gains: &[f64]) -> Vec<Vec<Complex64>> {
        let mut branches: Vec<Vec<Complex64>> = Vec::new();
        for (i, &k) in gains.iter().enumerate() {
            let mut roots = (&self.den + &self.num.scale(k)).roots();
            if i == 0 {
                branches = roots.into_iter().map(|r| vec![r]).collect();
                continue;
            }
            for branch in &mut branches {
                let last = *branch.last().unwrap();
                let nearest = (0..roots.len()).min_by(|&a, &b| {
                    (roots[a] - last).norm().total_cmp(&(roots[b] - last).norm())
                });
                branch.push(match nearest {
                    Some(j) => roots.swap_remove(j),
                    None => Complex64::new(f64::NAN, f64::NAN), // degree dropped at this gain
                });
            }
        }
        branches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_order_breakaway() {
        // L = 1/(s(s+2)): poles meet at -1 when k = 1, then split vertically.
        let branches = Tf::new([1.0], [1.0, 2.0, 0.0]).root_locus(&[0.0, 0.5, 5.0]);
        assert_eq!(branches.len(), 2);
        for b in &branches {
            assert!((b[2].re + 1.0).abs() < 1e-9);
            assert!((b[2].im.abs() - 2.0).abs() < 1e-9);
        }
        assert!(branches.iter().any(|b| b[0].re == -2.0) && branches.iter().any(|b| b[0].re == 0.0));
    }
}
