use crate::ss::{dot, mat_vec};
use crate::tf::Tf;

impl Tf {
    /// Simulates the response to input `u(t)` from rest, over `[0, t_end]` with
    /// fixed step `dt` (RK4 on the controllable canonical state-space form).
    /// Returns `(t, y)`. Panics if the transfer function is improper.
    pub fn simulate(&self, u: impl Fn(f64) -> f64, t_end: f64, dt: f64) -> (Vec<f64>, Vec<f64>) {
        let ss = self.to_ss();
        let deriv = |x: &[f64], u: f64| -> Vec<f64> {
            mat_vec(&ss.a, x).iter().zip(&ss.b).map(|(ax, b)| ax + b * u).collect()
        };
        let shifted = |x: &[f64], k: &[f64], h: f64| -> Vec<f64> {
            x.iter().zip(k).map(|(xi, ki)| xi + h * ki).collect()
        };

        let steps = (t_end / dt).round() as usize;
        let mut x = vec![0.0; ss.b.len()];
        let (mut ts, mut ys) = (Vec::with_capacity(steps + 1), Vec::with_capacity(steps + 1));
        for i in 0..=steps {
            let t = i as f64 * dt;
            ts.push(t);
            ys.push(dot(&ss.c, &x) + ss.d * u(t));

            let k1 = deriv(&x, u(t));
            let k2 = deriv(&shifted(&x, &k1, dt / 2.0), u(t + dt / 2.0));
            let k3 = deriv(&shifted(&x, &k2, dt / 2.0), u(t + dt / 2.0));
            let k4 = deriv(&shifted(&x, &k3, dt), u(t + dt));
            for j in 0..x.len() {
                x[j] += dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
        }
        (ts, ys)
    }

    /// Unit step response.
    pub fn step_response(&self, t_end: f64, dt: f64) -> (Vec<f64>, Vec<f64>) {
        self.simulate(|_| 1.0, t_end, dt)
    }

    /// A time horizon and step size that suit this system's pole locations:
    /// about ten time constants of the slowest pole, resolving the fastest.
    pub fn sim_grid(&self) -> (f64, f64) {
        let poles = self.poles();
        let slowest = poles.iter().map(|p| p.re.abs()).filter(|r| *r > 1e-9).fold(f64::INFINITY, f64::min);
        let fastest = poles.iter().map(|p| p.norm()).fold(1e-9, f64::max);
        let t_end = if slowest.is_finite() { (10.0 / slowest).clamp(1e-3, 1e5) } else { 10.0 };
        let dt = (t_end / 5000.0).min(0.1 / fastest).max(t_end / 1e6);
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
        assert!((y[100] - (1.0 - (-1.0_f64).exp())).abs() < 1e-8);
    }

    #[test]
    fn second_order_settles_to_dc_gain() {
        let g = Tf::new([1.0, 5.0], [1.0, 2.0, 5.0]); // (s + 5)/(s^2 + 2s + 5)
        let (_, y) = g.step_response(20.0, 0.001);
        assert!((y.last().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn biproper_has_direct_feedthrough() {
        let (_, y) = Tf::new([2.0, 1.0], [1.0, 1.0]).step_response(0.1, 0.01);
        assert_eq!(y[0], 2.0);
    }
}
