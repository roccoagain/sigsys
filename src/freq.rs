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

/// Gain margin (as a ratio, at the -180° phase crossover) and phase margin
/// (in degrees, at the 0 dB gain crossover). `None` when there is no crossing.
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

/// Finds a root of `f` between `a` and `b` (which bracket a sign change), in log-frequency.
fn bisect(mut a: f64, mut b: f64, f: impl Fn(f64) -> f64) -> f64 {
    let sign_a = f(a) >= 0.0;
    for _ in 0..60 {
        let m = (a * b).sqrt();
        if (f(m) >= 0.0) == sign_a {
            a = m;
        } else {
            b = m;
        }
    }
    (a * b).sqrt()
}

impl Tf {
    /// Two decades either side of the nonzero poles and zeros.
    pub fn freq_range(&self) -> (f64, f64) {
        let mags: Vec<f64> =
            self.poles().iter().chain(&self.zeros()).map(|z| z.norm()).filter(|m| *m > 1e-9).collect();
        if mags.is_empty() {
            return (0.01, 100.0);
        }
        let lo = mags.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = mags.iter().copied().fold(0.0, f64::max);
        (lo / 100.0, hi * 100.0)
    }

    /// Magnitude and unwrapped phase at `n` log-spaced frequencies.
    pub fn bode_sweep(&self, w_min: f64, w_max: f64, n: usize) -> Vec<BodePoint> {
        // Integrators contribute -90° each at low frequency; start the unwrap there.
        let mut prev = -90.0 * self.system_type() as f64;
        logspace(w_min, w_max, n)
            .into_iter()
            .map(|w| {
                let g = self.freq_response(w);
                prev = unwrap_near(g.arg().to_degrees(), prev);
                BodePoint { w, mag_db: 20.0 * g.norm().log10(), phase_deg: prev }
            })
            .collect()
    }

    /// Gain and phase margins of `self` as an open loop, using the first
    /// crossing of each kind over [`Tf::freq_range`].
    pub fn margins(&self) -> Margins {
        let (lo, hi) = self.freq_range();
        let pts = self.bode_sweep(lo, hi, 4000);
        // Crossings are tested with a small tolerance so a response sitting at
        // exactly 0 dB (e.g. a unity-gain filter's passband) doesn't count.
        let mag_db = |w: f64| 20.0 * self.freq_response(w).norm().log10();

        let phase = pts.windows(2).find(|p| (p[0].mag_db > 1e-9) != (p[1].mag_db > 1e-9)).map(|p| {
            let w = bisect(p[0].w, p[1].w, mag_db);
            let phase = unwrap_near(self.freq_response(w).arg().to_degrees(), p[0].phase_deg);
            Crossing { w, margin: 180.0 + phase }
        });

        let gain = pts
            .windows(2)
            .find(|p| (p[0].phase_deg > -180.0) != (p[1].phase_deg > -180.0))
            .map(|p| {
                let reference = p[0].phase_deg;
                let w = bisect(p[0].w, p[1].w, |w| {
                    unwrap_near(self.freq_response(w).arg().to_degrees(), reference) + 180.0
                });
                Crossing { w, margin: 1.0 / self.freq_response(w).norm() }
            });

        Margins { gain, phase }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn phase_unwraps_past_180() {
        let pts = Tf::new([1.0], [1.0, 3.0, 3.0, 1.0]).bode_sweep(0.01, 100.0, 200); // (s+1)^-3
        assert!((pts.last().unwrap().phase_deg + 270.0).abs() < 2.0);
    }
}
