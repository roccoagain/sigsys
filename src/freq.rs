use num_complex::Complex64;

use crate::poly::Poly;
use crate::signals::logspace;
use crate::tf::Tf;

/// One point of a Bode sweep. Phase is unwrapped (continuous across ±180°).
#[derive(Clone, Copy, Debug)]
pub struct BodePoint {
    pub w: f64,
    pub mag_db: f64,
    pub phase_deg: f64,
}

/// A stability margin and the frequency (rad/s) where it is measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossing {
    pub w: f64,
    pub margin: f64,
}

/// Gain margin (as a ratio, at a phase crossover where `L(jω)` is real and
/// negative) and phase margin (in degrees, wrapped to (-180, 180], at a 0 dB gain
/// crossover). When there are several crossings, each is the one closest to
/// instability. `None` when there is no crossing.
#[derive(Clone, Copy, Debug)]
pub struct Margins {
    pub gain: Option<Crossing>,
    pub phase: Option<Crossing>,
}

impl Margins {
    pub fn gain_margin_db(&self) -> Option<f64> {
        self.gain.map(|c| 20.0 * c.margin.log10())
    }
}

/// Shifts `deg` by a multiple of 360° to land within 180° of `reference`.
fn unwrap_near(deg: f64, reference: f64) -> f64 {
    deg - 360.0 * ((deg - reference) / 360.0).round()
}

fn wrap_180(deg: f64) -> f64 {
    let d = deg.rem_euclid(360.0);
    if d > 180.0 { d - 360.0 } else { d }
}

/// Real and imaginary parts of `p(jω)` as real polynomials in `ω`.
fn on_imag_axis(p: &Poly) -> (Poly, Poly) {
    let n = p.degree();
    let (mut re, mut im) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    for (i, &c) in p.coeffs().iter().enumerate() {
        match (n - i) % 4 {
            0 => re[i] = c,  // j^0 = 1
            1 => im[i] = c,  // j^1 = j
            2 => re[i] = -c, // j^2 = -1
            _ => im[i] = -c, // j^3 = -j
        }
    }
    (Poly::new(re), Poly::new(im))
}

/// Nonnegative real roots, deduplicated.
fn nonnegative_real_roots(p: &Poly) -> Vec<f64> {
    if p.is_zero() {
        return vec![];
    }
    let mut ws: Vec<f64> = p
        .roots()
        .iter()
        .filter(|z| z.im.abs() <= 1e-6 * z.re.abs().max(1e-3) && z.re >= -1e-12)
        .map(|z| z.re.max(0.0))
        .collect();
    ws.sort_by(f64::total_cmp);
    ws.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * b.abs().max(1.0));
    ws
}

impl Tf {
    /// Two decades either side of the nonzero poles, zeros, and margin crossovers.
    pub fn freq_range(&self) -> (f64, f64) {
        let m = self.margins();
        let mags: Vec<f64> = self
            .poles()
            .iter()
            .chain(&self.zeros())
            .map(|z| z.norm())
            .chain([m.gain, m.phase].into_iter().flatten().map(|c| c.w))
            .filter(|w| *w > 1e-9)
            .collect();
        if mags.is_empty() {
            return (0.01, 100.0);
        }
        let lo = mags.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = mags.iter().copied().fold(0.0, f64::max);
        (lo / 100.0, hi * 100.0)
    }

    /// Phase (degrees) of the low-frequency asymptote `k s^(nz - dz)`.
    fn low_freq_phase(&self) -> f64 {
        if self.num.is_zero() {
            return 0.0;
        }
        let (nz, dz) = (self.num.zeros_at_origin(), self.den.zeros_at_origin());
        let (n, d) = (self.num.coeffs(), self.den.coeffs());
        let sign = if n[n.len() - 1 - nz] / d[d.len() - 1 - dz] < 0.0 { -180.0 } else { 0.0 };
        sign + 90.0 * (nz as f64 - dz as f64)
    }

    /// Magnitude and unwrapped phase at `n` log-spaced frequencies. Phase starts
    /// from the low-frequency asymptote; across a jω-axis pole or zero the true
    /// phase jumps by ±180° and which way it is unwrapped is arbitrary.
    pub fn bode_sweep(&self, w_min: f64, w_max: f64, n: usize) -> Vec<BodePoint> {
        let mut prev = self.low_freq_phase();
        logspace(w_min, w_max, n)
            .into_iter()
            .map(|w| {
                let g = self.freq_response(w);
                prev = unwrap_near(g.arg().to_degrees(), prev);
                BodePoint { w, mag_db: 20.0 * g.norm().log10(), phase_deg: prev }
            })
            .collect()
    }

    /// Gain and phase margins of `self` as an open loop. Crossovers are found
    /// exactly, as roots of `|N(jω)|² - |D(jω)|²` (gain crossover) and
    /// `Im(N(jω) conj(D(jω)))` (phase crossover), so none are missed.
    pub fn margins(&self) -> Margins {
        let (nr, ni) = on_imag_axis(&self.num);
        let (dr, di) = on_imag_axis(&self.den);
        let gain_poly = &(&(&nr * &nr) + &(&ni * &ni)) - &(&(&dr * &dr) + &(&di * &di));
        let phase_poly = &(&ni * &dr) - &(&nr * &di);

        // Only count roots where |L| - 1 changes sign: touching 0 dB (e.g. a unity-gain
        // filter's passband at ω = 0) is not a crossover.
        let g = |w: f64| gain_poly.eval(Complex64::new(w, 0.0)).re;
        let phase = nonnegative_real_roots(&gain_poly)
            .into_iter()
            .filter(|w| *w > 0.0 && g(w * (1.0 - 1e-6)) * g(w * (1.0 + 1e-6)) < 0.0)
            .map(|w| (w, self.freq_response(w)))
            .filter(|(_, l)| l.is_finite())
            .map(|(w, l)| Crossing { w, margin: wrap_180(180.0 + l.arg().to_degrees()) })
            .min_by(|a, b| a.margin.total_cmp(&b.margin));

        let gain = nonnegative_real_roots(&phase_poly)
            .into_iter()
            .map(|w| (w, self.freq_response(w)))
            .filter(|(_, l)| l.is_finite() && l.re < 0.0 && l.norm() < 1e8) // skip jω-axis poles
            .map(|(w, l)| Crossing { w, margin: 1.0 / l.norm() })
            .min_by(|a, b| a.margin.ln().abs().total_cmp(&b.margin.ln().abs()));

        Margins { gain, phase }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_amp_margins_survive_wide_coefficient_range() {
        let pm = Tf::new([1e7], [1.0, 1.0]).margins().phase.unwrap();
        assert!((pm.w - 1e7).abs() < 1.0 && (pm.margin - 90.0).abs() < 1e-3, "{pm:?}");
    }

    #[test]
    fn classic_third_order_margins() {
        // L = 1 / (s(s+1)(s+2)): phase crossover at √2 with |L| = 1/6.
        let m = Tf::new([1.0], [1.0, 3.0, 2.0, 0.0]).margins();
        let gm = m.gain.unwrap();
        assert!((gm.w - 2f64.sqrt()).abs() < 1e-6);
        assert!((gm.margin - 6.0).abs() < 1e-6);
        let pm = m.phase.unwrap();
        assert!((pm.w - 0.4457).abs() < 1e-3);
        assert!((pm.margin - 53.4).abs() < 0.1);
    }

    #[test]
    fn first_order_has_infinite_gain_margin() {
        let m = Tf::new([10.0], [1.0, 1.0]).margins();
        assert!(m.gain.is_none());
        assert!((m.phase.unwrap().margin - (180.0 - 84.26)).abs() < 0.01);
    }

    #[test]
    fn flat_0db_passband_is_not_a_crossover() {
        assert!(Tf::butterworth(4, 2.0).margins().phase.is_none());
    }

    #[test]
    fn crossovers_far_from_poles() {
        let m = Tf::new([1e6], [1.0, 1.0]).margins().phase.unwrap();
        assert!((m.w - 1e6).abs() < 1.0 && (m.margin - 90.0).abs() < 1e-3);
        let m = Tf::new([1000.0], [1.0, 0.0]).margins().phase.unwrap();
        assert!((m.w - 1000.0).abs() < 1e-6 && (m.margin - 90.0).abs() < 1e-9);
    }

    #[test]
    fn negative_gain() {
        // -10/(s+1): closed loop pole at +9, so both margins must say unstable.
        let m = Tf::new([-10.0], [1.0, 1.0]).margins();
        assert!((m.phase.unwrap().margin + 84.26).abs() < 0.01);
        let gm = m.gain.unwrap();
        assert_eq!(gm.w, 0.0);
        assert!((gm.margin - 0.1).abs() < 1e-12);
    }

    #[test]
    fn undamped_poles_give_negative_phase_margin() {
        // 1/((s^2+4)(s+1)) closes to an unstable loop.
        let m = Tf::new([1.0], [1.0, 1.0, 4.0, 4.0]).margins();
        assert!(m.phase.unwrap().margin < 0.0);
    }

    #[test]
    fn low_frequency_phase_includes_zeros_and_sign() {
        let pts = Tf::new([1.0, 0.0, 0.0, 0.0], [1.0, 3.0, 3.0, 1.0]).bode_sweep(1e-3, 1e-2, 3);
        assert!((pts[0].phase_deg - 270.0).abs() < 1.0);
        let pts = Tf::new([-1.0], [1.0, 1.0]).bode_sweep(1e-3, 1e-2, 3);
        assert!((pts[0].phase_deg + 180.0).abs() < 1.0);
    }

    #[test]
    fn phase_unwraps_past_180() {
        let pts = Tf::new([1.0], [1.0, 3.0, 3.0, 1.0]).bode_sweep(0.01, 100.0, 200); // (s+1)^-3
        assert!((pts.last().unwrap().phase_deg + 270.0).abs() < 2.0);
    }
}
