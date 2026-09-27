use crate::tf::Tf;

/// Step response characteristics. Rise time is 10%–90%, settling is to within 2%.
#[derive(Clone, Copy, Debug)]
pub struct StepInfo {
    pub rise_time: f64,
    pub settling_time: f64,
    pub peak_time: f64,
    pub peak: f64,
    pub overshoot_pct: f64,
    pub final_value: f64,
}

impl Tf {
    /// `None` if the system is unstable or improper, its final value is zero, or the simulated
    /// response fails to rise through 90% and settle within 2% of the final value.
    pub fn step_info(&self) -> Option<StepInfo> {
        if !self.is_stable() {
            return None;
        }
        let final_value = self.dc_gain();
        if final_value == 0.0 || !final_value.is_finite() {
            return None;
        }
        // sim_grid's horizon is ten time constants of the slowest pole, but repeated or
        // clustered slow poles settle later, so lengthen it (up to 8x) until they do.
        let (t_end, dt) = self.sim_grid();
        let mut scale = 1.0;
        let (t, y) = loop {
            let (t, y) = self.step_response(scale * t_end, dt).ok()?;
            if scale >= 8.0 || y.last().is_some_and(|v| (v / final_value - 1.0).abs() <= 0.02) {
                break (t, y);
            }
            scale *= 2.0;
        };
        if y.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let r: Vec<f64> = y.iter().map(|v| v / final_value).collect();

        // First time the normalized response reaches `level`, linearly interpolated.
        let cross = |level: f64| match r.iter().position(|v| *v >= level) {
            Some(0) => t[0],
            Some(i) => t[i - 1] + (level - r[i - 1]) / (r[i] - r[i - 1]) * dt,
            None => f64::NAN,
        };
        let (i_peak, r_peak) = r.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        let rise_time = cross(0.9) - cross(0.1);
        let unsettled = |v: &f64| (v - 1.0).abs() > 0.02;
        if rise_time.is_nan() || r.last().is_some_and(unsettled) {
            return None;
        }
        let settling_time = r.iter().rposition(unsettled).map_or(0.0, |i| t[i + 1]);

        Some(StepInfo {
            rise_time,
            settling_time,
            peak_time: t[i_peak],
            peak: y[i_peak],
            overshoot_pct: ((r_peak - 1.0) * 100.0).max(0.0),
            final_value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn very_slow_pole() {
        let info = Tf::new([1e-5], [1.0, 1e-5]).step_info().unwrap();
        assert!((info.rise_time - 9f64.ln() * 1e5).abs() < 1e2, "{info:?}");
        assert!((info.final_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn first_order() {
        let info = Tf::new([3.0], [1.0, 1.0]).step_info().unwrap();
        assert!((info.rise_time - 9f64.ln()).abs() < 1e-3);
        assert!((info.settling_time - 50f64.ln()).abs() < 1e-2);
        assert_eq!(info.overshoot_pct, 0.0);
        assert!((info.final_value - 3.0).abs() < 1e-12);
    }

    #[test]
    fn second_order_underdamped() {
        // ω_n = 1, ζ = 0.5: overshoot e^(-πζ/√(1-ζ²)), peak at π/ω_d.
        let info = Tf::new([1.0], [1.0, 1.0, 1.0]).step_info().unwrap();
        let wd = 0.75_f64.sqrt();
        let os = (-std::f64::consts::PI * 0.5 / wd).exp() * 100.0;
        assert!((info.overshoot_pct - os).abs() < 0.01);
        assert!((info.peak_time - std::f64::consts::PI / wd).abs() < 0.01);
    }

    #[test]
    fn stiff_system() {
        // Slow pole at -1e-3 dominates: rise ≈ ln 9 / 1e-3 ≈ 2197 s.
        let info = Tf::new([10.0], [1.0, 1e4 + 1e-3, 10.0]).step_info().unwrap();
        assert!((info.rise_time - 2197.2).abs() < 1.0, "{info:?}");
        assert!(info.peak.is_finite());
    }

    #[test]
    fn repeated_poles_still_settle() {
        // (s+1)^-n settles later than ten time constants once n >= 5; step_info
        // used to give up and return None for these stable systems.
        for n in 1..=10 {
            let g = Tf::from_polys(crate::Poly::new([1.0]), crate::Poly::new([1.0, 1.0]).pow(n));
            let info = g.step_info().unwrap_or_else(|| panic!("n = {n}"));
            assert!((info.final_value - 1.0).abs() < 1e-12 && info.overshoot_pct == 0.0);
        }
    }

    #[test]
    fn unstable_has_no_info() {
        assert!(Tf::new([1.0], [1.0, -1.0]).step_info().is_none());
    }
}
