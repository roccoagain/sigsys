use num_complex::Complex64;

use crate::tf::Tf;

impl Tf {
    /// Closed-loop poles of `1 + k L(s) = 0` for each gain, where `self` is `L(s)`.
    /// Poles are grouped into continuous branches: `branches[b][i]` is branch `b`
    /// at `gains[i]`. There is one branch per closed-loop pole in general
    /// (`max(deg num, deg den)`); at gains where the degree drops, the missing
    /// poles are NaN.
    pub fn root_locus(&self, gains: &[f64]) -> Vec<Vec<Complex64>> {
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let n = self.den.degree().max(self.num.degree());
        let mut branches: Vec<Vec<Complex64>> = vec![Vec::with_capacity(gains.len()); n];
        let mut last: Vec<Option<Complex64>> = vec![None; n];
        for &k in gains {
            let roots = (&self.den + &self.num.scale(k)).roots();
            // Match globally closest (branch, root) pairs first, so branches that
            // pass near each other don't steal each other's poles.
            let mut pairs: Vec<(f64, usize, usize)> = (0..n)
                .flat_map(|b| (0..roots.len()).map(move |r| (b, r)))
                .map(|(b, r)| (last[b].map_or(f64::MAX, |z| (roots[r] - z).norm()), b, r))
                .collect();
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut assigned = vec![None; n];
            let mut used = vec![false; roots.len()];
            for (_, b, r) in pairs {
                if assigned[b].is_none() && !used[r] {
                    assigned[b] = Some(roots[r]);
                    used[r] = true;
                }
            }
            for (b, root) in assigned.into_iter().enumerate() {
                branches[b].push(root.unwrap_or(nan));
                if root.is_some() {
                    last[b] = root;
                }
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

    #[test]
    fn improper_keeps_every_branch() {
        // (s+1)^2/(s+1): closed loop (s+1) + k(s+1)^2 has two poles for k > 0.
        let branches = Tf::new([1.0, 2.0, 1.0], [1.0, 1.0]).root_locus(&[0.0, 1.0]);
        assert_eq!(branches.len(), 2);
        assert!(branches.iter().all(|b| b[1].re.is_finite()));
    }

    #[test]
    fn degree_drop_is_nan_not_empty() {
        // (s+2)/(s+1) at k = -1: (s+1) - (s+2) = -1 has no roots.
        let branches = Tf::new([1.0, 2.0], [1.0, 1.0]).root_locus(&[-1.0, 0.0, 1.0]);
        assert_eq!(branches.len(), 1);
        assert!(branches[0][0].re.is_nan());
        assert_eq!(branches[0][1].re, -1.0);
        assert_eq!(branches[0][2].re, -1.5);
    }
}
