use crate::ss::{discretize, dot, mat_vec};
use crate::tf::Tf;

impl Tf {
    /// Simulates the response to input `u(t)` from rest, over `[0, t_end]` with
    /// step `dt`. The state update is exact when `u` is linear between samples
    /// (so exact for steps and ramps) and stays stable for any `dt`, however
    /// stiff the system. Returns `(t, y)`. Panics if the transfer function is improper.
    pub fn simulate(&self, u: impl Fn(f64) -> f64, t_end: f64, dt: f64) -> (Vec<f64>, Vec<f64>) {
        let ss = self.to_ss();
        let (phi, gamma1, gamma2) = discretize(&ss, dt);
        let steps = (t_end / dt).round() as usize;
        let mut x = vec![0.0; ss.b.len()];
        let (mut ts, mut ys) = (Vec::with_capacity(steps + 1), Vec::with_capacity(steps + 1));
        for i in 0..=steps {
            let t = i as f64 * dt;
            let (u_now, u_next) = (u(t), u(t + dt));
            ts.push(t);
            ys.push(dot(&ss.c, &x) + ss.d * u_now);
            x = mat_vec(&phi, &x)
                .iter()
                .zip(gamma1.iter().zip(&gamma2))
                .map(|(px, (g1, g2))| px + g1 * u_now + g2 * (u_next - u_now))
                .collect();
        }
        (ts, ys)
    }

    /// Unit step response.
    pub fn step_response(&self, t_end: f64, dt: f64) -> (Vec<f64>, Vec<f64>) {
        self.simulate(|_| 1.0, t_end, dt)
    }

    /// A time horizon and step size that suit this system's pole locations:
    /// about ten time constants of the slowest pole, resolving the fastest
    /// where that takes no more than 200,000 steps.
    pub fn sim_grid(&self) -> (f64, f64) {
        let poles = self.poles();
        let slowest = poles.iter().map(|p| p.re.abs()).filter(|r| *r > 1e-9).fold(f64::INFINITY, f64::min);
        let fastest = poles.iter().map(|p| p.norm()).fold(1e-9, f64::max);
        let t_end = if slowest.is_finite() { (10.0 / slowest).clamp(1e-3, 1e5) } else { 10.0 };
        let dt = (t_end / 5000.0).min(0.1 / fastest).max(t_end / 2e5);
        (t_end, dt)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tf;

    #[test]
    fn first_order_step() {
        let (t, y) = Tf::new([1.0], [1.0, 1.0]).step_response(1.0, 0.01);
        assert_eq!(t.len(), 101);
        assert!((y[100] - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn ramp_is_exact() {
        // 1/(s+1) driven by u = t: y = t - 1 + e^-t
        let (t, y) = Tf::new([1.0], [1.0, 1.0]).simulate(|t| t, 3.0, 0.1);
        for (t, y) in t.iter().zip(&y) {
            assert!((y - (t - 1.0 + (-t).exp())).abs() < 1e-12);
        }
    }

    #[test]
    fn second_order_settles_to_dc_gain() {
        let g = Tf::new([1.0, 5.0], [1.0, 2.0, 5.0]); // (s + 5)/(s^2 + 2s + 5)
        let (_, y) = g.step_response(20.0, 0.001);
        assert!((y.last().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stiff_system_stays_bounded() {
        let g = Tf::new([10.0], [1.0, 1e4 + 1e-3, 10.0]); // poles at -1e-3 and -1e4
        let (t_end, dt) = g.sim_grid();
        let (_, y) = g.step_response(t_end, dt);
        assert!(y.iter().all(|v| v.is_finite() && *v <= 1.0 + 1e-9));
    }

    #[test]
    fn biproper_has_direct_feedthrough() {
        let (_, y) = Tf::new([2.0, 1.0], [1.0, 1.0]).step_response(0.1, 0.01);
        assert_eq!(y[0], 2.0);
    }
}
