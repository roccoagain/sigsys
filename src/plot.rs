//! Minimal SVG line/scatter plots, plus ready-made plots for transfer functions.

use std::fmt::Write as _;
use std::{fs, io, path::Path};

use num_complex::Complex64;

use crate::discrete::Dtf;
use crate::signals::{linspace, logspace};
use crate::tf::Tf;

const WIDTH: f64 = 720.0;
const HEIGHT: f64 = 380.0;
const COLORS: [&str; 6] = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"];

#[derive(Clone, Copy, Debug)]
pub enum Marker {
    Dot,
    Cross,
    Circle,
}

#[derive(Clone, Copy, Debug)]
pub enum Style {
    Line,
    Markers(Marker),
}

#[derive(Clone, Debug)]
pub struct Series {
    pub label: String,
    pub points: Vec<(f64, f64)>,
    pub style: Style,
}

/// A single 2-D plot. Build it with the chained methods, then `save` it.
#[derive(Clone, Debug, Default)]
pub struct Plot {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub log_x: bool,
    pub series: Vec<Series>,
    /// Dashed reference lines.
    pub hlines: Vec<f64>,
    pub vlines: Vec<f64>,
}

impl Plot {
    pub fn new(title: &str) -> Self {
        Plot { title: title.into(), ..Default::default() }
    }

    pub fn labels(mut self, x: &str, y: &str) -> Self {
        self.x_label = x.into();
        self.y_label = y.into();
        self
    }

    pub fn log_x(mut self) -> Self {
        self.log_x = true;
        self
    }

    pub fn line(mut self, label: &str, points: Vec<(f64, f64)>) -> Self {
        self.series.push(Series { label: label.into(), points, style: Style::Line });
        self
    }

    pub fn markers(mut self, label: &str, points: Vec<(f64, f64)>, marker: Marker) -> Self {
        self.series.push(Series { label: label.into(), points, style: Style::Markers(marker) });
        self
    }

    pub fn hline(mut self, y: f64) -> Self {
        self.hlines.push(y);
        self
    }

    pub fn vline(mut self, x: f64) -> Self {
        self.vlines.push(x);
        self
    }

    pub fn to_svg(&self) -> String {
        svg(&[self])
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        fs::write(path, self.to_svg())
    }

    fn render(&self, out: &mut String, oy: f64, id: usize) {
        let (left, right, top, bottom) = (70.0, 20.0, 36.0, 48.0);
        let (x0, y0, pw, ph) = (left, oy + top, WIDTH - left - right, HEIGHT - top - bottom);
        let tx = |x: f64| if self.log_x { x.log10() } else { x };
        let usable = |x: f64, y: f64| x.is_finite() && y.is_finite() && (!self.log_x || x > 0.0);

        let points = self.series.iter().flat_map(|s| &s.points).filter(|(x, y)| usable(*x, *y));
        let xs: Vec<f64> = points
            .clone()
            .map(|p| tx(p.0))
            .chain(self.vlines.iter().filter(|x| usable(**x, 0.0)).map(|x| tx(*x)))
            .collect();
        let ys: Vec<f64> = points.map(|p| p.1).chain(self.hlines.iter().copied()).collect();
        // Pad x only for scatter plots, so markers at the extremes aren't clipped.
        let has_markers = self.series.iter().any(|s| matches!(s.style, Style::Markers(_)));
        let (xmin, xmax) = range(&xs, if has_markers && !self.log_x { 0.05 } else { 0.0 });
        let (ymin, ymax) = range(&ys, 0.05);
        let px = |x: f64| x0 + (tx(x) - xmin) / (xmax - xmin) * pw;
        let py = |y: f64| (y0 + ph - (y - ymin) / (ymax - ymin) * ph).clamp(-1e6, 1e6);

        // Grid and tick labels.
        let xticks = if self.log_x {
            log_ticks(xmin, xmax)
        } else {
            let (step, ticks) = nice_ticks(xmin, xmax, false);
            ticks.into_iter().map(|v| (v, fmt_num(v, step))).collect()
        };
        for (v, label) in xticks {
            let x = x0 + (v - xmin) / (xmax - xmin) * pw;
            write!(out, r##"<line x1="{x:.1}" y1="{y0:.1}" x2="{x:.1}" y2="{:.1}" stroke="#e5e7eb"/>"##, y0 + ph).unwrap();
            write!(out, r#"<text x="{x:.1}" y="{:.1}" text-anchor="middle">{label}</text>"#, y0 + ph + 16.0).unwrap();
        }
        let (ystep, yticks) = nice_ticks(ymin, ymax, false);
        for v in yticks {
            let y = py(v);
            write!(out, r##"<line x1="{x0:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="#e5e7eb"/>"##, x0 + pw).unwrap();
            write!(out, r#"<text x="{:.1}" y="{:.1}" text-anchor="end">{}</text>"#, x0 - 6.0, y + 4.0, fmt_num(v, ystep))
                .unwrap();
        }

        // Titles.
        let cx = x0 + pw / 2.0;
        let cy = y0 + ph / 2.0;
        write!(out, r#"<text x="{cx:.1}" y="{:.1}" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#, oy + 22.0, esc(&self.title)).unwrap();
        write!(out, r#"<text x="{cx:.1}" y="{:.1}" text-anchor="middle">{}</text>"#, oy + HEIGHT - 10.0, esc(&self.x_label)).unwrap();
        write!(out, r#"<text x="18" y="{cy:.1}" text-anchor="middle" transform="rotate(-90 18 {cy:.1})">{}</text>"#, esc(&self.y_label)).unwrap();

        // Data, clipped to the plot area.
        write!(out, r#"<clipPath id="clip{id}"><rect x="{x0}" y="{y0}" width="{pw}" height="{ph}"/></clipPath><g clip-path="url(#clip{id})">"#).unwrap();
        for &y in &self.hlines {
            write!(out, r##"<line x1="{x0:.1}" y1="{0:.1}" x2="{1:.1}" y2="{0:.1}" stroke="#6b7280" stroke-dasharray="4 3"/>"##, py(y), x0 + pw).unwrap();
        }
        for &x in self.vlines.iter().filter(|x| usable(**x, 0.0)) {
            write!(out, r##"<line x1="{0:.1}" y1="{y0:.1}" x2="{0:.1}" y2="{1:.1}" stroke="#6b7280" stroke-dasharray="4 3"/>"##, px(x), y0 + ph).unwrap();
        }
        for (i, s) in self.series.iter().enumerate() {
            let color = COLORS[i % COLORS.len()];
            match s.style {
                Style::Line => {
                    // Break the line wherever a point can't be drawn.
                    for segment in s.points.split(|(x, y)| !usable(*x, *y)).filter(|seg| seg.len() > 1) {
                        let pts: Vec<String> = segment.iter().map(|(x, y)| format!("{:.2},{:.2}", px(*x), py(*y))).collect();
                        write!(out, r#"<polyline points="{}" fill="none" stroke="{color}" stroke-width="2"/>"#, pts.join(" ")).unwrap();
                    }
                }
                Style::Markers(m) => {
                    for &(x, y) in s.points.iter().filter(|(x, y)| usable(*x, *y)) {
                        marker(out, m, px(x), py(y), color);
                    }
                }
            }
        }
        out.push_str("</g>");
        write!(out, r##"<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#374151"/>"##).unwrap();

        // Legend, top-right inside the plot.
        let labeled: Vec<(usize, &Series)> = self.series.iter().enumerate().filter(|(_, s)| !s.label.is_empty()).collect();
        if !labeled.is_empty() {
            let longest = labeled.iter().map(|(_, s)| s.label.chars().count()).max().unwrap();
            let (w, h) = (40.0 + longest as f64 * 7.0, 8.0 + 18.0 * labeled.len() as f64);
            // Use whichever corner covers the fewest data points.
            let drawn: Vec<(f64, f64)> = self
                .series
                .iter()
                .flat_map(|s| &s.points)
                .filter(|(x, y)| usable(*x, *y))
                .map(|(x, y)| (px(*x), py(*y)))
                .collect();
            let corners = [
                (x0 + pw - w - 8.0, y0 + 8.0),
                (x0 + 8.0, y0 + 8.0),
                (x0 + pw - w - 8.0, y0 + ph - h - 8.0),
                (x0 + 8.0, y0 + ph - h - 8.0),
            ];
            let covered = |(cx, cy): (f64, f64)| {
                drawn.iter().filter(|(x, y)| *x >= cx - 6.0 && *x <= cx + w + 6.0 && *y >= cy - 6.0 && *y <= cy + h + 6.0).count()
            };
            let (lx, ly) = corners.into_iter().min_by_key(|c| covered(*c)).unwrap();
            write!(out, r##"<rect x="{lx:.1}" y="{ly:.1}" width="{w:.1}" height="{h:.1}" fill="white" fill-opacity="0.9" stroke="#d1d5db"/>"##).unwrap();
            for (row, (i, s)) in labeled.iter().enumerate() {
                let color = COLORS[i % COLORS.len()];
                let y = ly + 16.0 + 18.0 * row as f64;
                match s.style {
                    Style::Line => write!(out, r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{color}" stroke-width="2"/>"#, lx + 8.0, y - 4.0, lx + 28.0, y - 4.0).unwrap(),
                    Style::Markers(m) => marker(out, m, lx + 18.0, y - 4.0, color),
                }
                write!(out, r#"<text x="{:.1}" y="{y:.1}">{}</text>"#, lx + 34.0, esc(&s.label)).unwrap();
            }
        }
    }
}

/// Saves several plots stacked vertically in one SVG (e.g. a Bode pair).
pub fn save_stacked(plots: &[Plot], path: impl AsRef<Path>) -> io::Result<()> {
    fs::write(path, svg(&plots.iter().collect::<Vec<_>>()))
}

fn svg(plots: &[&Plot]) -> String {
    let total = HEIGHT * plots.len() as f64;
    let mut out = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{total}" viewBox="0 0 {WIDTH} {total}" font-family="Helvetica, Arial, sans-serif" font-size="12"><rect width="100%" height="100%" fill="white"/>"#
    );
    for (i, p) in plots.iter().enumerate() {
        p.render(&mut out, i as f64 * HEIGHT, i);
    }
    out.push_str("</svg>\n");
    out
}

fn marker(out: &mut String, m: Marker, x: f64, y: f64, color: &str) {
    match m {
        Marker::Dot => write!(out, r#"<circle cx="{x:.2}" cy="{y:.2}" r="2.5" fill="{color}"/>"#),
        Marker::Circle => write!(out, r#"<circle cx="{x:.2}" cy="{y:.2}" r="5" fill="white" stroke="{color}" stroke-width="2"/>"#),
        Marker::Cross => write!(
            out,
            r#"<path d="M{:.2} {:.2}L{:.2} {:.2}M{:.2} {:.2}L{:.2} {:.2}" stroke="{color}" stroke-width="2"/>"#,
            x - 5.0, y - 5.0, x + 5.0, y + 5.0, x - 5.0, y + 5.0, x + 5.0, y - 5.0
        ),
    }
    .unwrap();
}

fn range(v: &[f64], pad: f64) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 1.0);
    }
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if hi - lo <= 1e-12 * lo.abs().max(1.0) {
        return (lo - 1.0, hi + 1.0);
    }
    let p = (hi - lo) * pad;
    (lo - p, hi + p)
}

/// Tick step of 1, 2, or 5 × 10^k giving about six ticks (whole decades if `decades`).
fn nice_ticks(lo: f64, hi: f64, decades: bool) -> (f64, Vec<f64>) {
    let raw = (hi - lo) / 6.0;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let mut step = mag * if norm < 1.5 { 1.0 } else if norm < 3.0 { 2.0 } else if norm < 7.0 { 5.0 } else { 10.0 };
    if decades {
        step = step.max(1.0).round();
    }
    let first = (lo / step).ceil() as i64;
    let last = (hi / step + 1e-9).floor() as i64;
    (step, (first..=last).map(|i| i as f64 * step).collect())
}

/// Ticks for a log axis spanning `lo..hi` decades, as `(log10 position, label)`:
/// whole decades, or round linear values when fewer than two decades fall inside.
fn log_ticks(lo: f64, hi: f64) -> Vec<(f64, String)> {
    let (_, decades) = nice_ticks(lo, hi, true);
    if decades.len() >= 2 {
        return decades.into_iter().map(|v| (v, fmt_num(10f64.powf(v), 10f64.powf(v)))).collect();
    }
    let (step, ticks) = nice_ticks(10f64.powf(lo), 10f64.powf(hi), false);
    ticks.into_iter().map(|x| (x.log10(), fmt_num(x, step))).collect()
}

fn fmt_num(v: f64, step: f64) -> String {
    if v.abs() < step * 1e-6 {
        return "0".into();
    }
    let decimals = if step >= 1.0 { 0 } else { (-step.log10() - 1e-9).ceil() as usize };
    format!("{v:.decimals$}")
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn re_im(zs: &[Complex64]) -> Vec<(f64, f64)> {
    zs.iter().map(|z| (z.re, z.im)).collect()
}

impl Tf {
    /// Step response over an automatic horizon, with the final value marked if stable.
    pub fn step_plot(&self) -> Plot {
        let (t_end, dt) = self.sim_grid();
        let (t, y) = self.step_response(t_end, dt);
        let stride = (t.len() / 2000).max(1);
        let pts = t.iter().zip(&y).step_by(stride).map(|(t, y)| (*t, *y)).collect();
        let plot = Plot::new("Step response").labels("Time (s)", "Output").line("y(t)", pts);
        if self.is_stable() { plot.hline(self.dc_gain()) } else { plot }
    }

    /// Magnitude and phase plots, with gain/phase crossovers marked.
    /// Save together with [`save_stacked`].
    pub fn bode_plot(&self) -> [Plot; 2] {
        let (lo, hi) = self.freq_range();
        let pts = self.bode_sweep(lo, hi, 600);
        let mut mag = Plot::new("Bode magnitude")
            .labels("Frequency (rad/s)", "Magnitude (dB)")
            .log_x()
            .line("", pts.iter().map(|p| (p.w, p.mag_db)).collect())
            .hline(0.0);
        let mut phase = Plot::new("Bode phase")
            .labels("Frequency (rad/s)", "Phase (deg)")
            .log_x()
            .line("", pts.iter().map(|p| (p.w, p.phase_deg)).collect())
            .hline(-180.0);
        let m = self.margins();
        for c in [m.gain, m.phase].into_iter().flatten() {
            mag = mag.vline(c.w);
            phase = phase.vline(c.w);
        }
        [mag, phase]
    }

    /// Poles (×) and zeros (○) in the s-plane.
    pub fn pzmap_plot(&self) -> Plot {
        let mut p = Plot::new("Pole-zero map").labels("Real", "Imaginary").hline(0.0).vline(0.0);
        p = p.markers("poles", re_im(&self.poles()), Marker::Cross);
        let zeros = self.zeros();
        if !zeros.is_empty() {
            p = p.markers("zeros", re_im(&zeros), Marker::Circle);
        }
        p
    }

    /// Root locus of `self` as an open loop, for gains 0..=`k_max`.
    pub fn root_locus_plot(&self, k_max: f64) -> Plot {
        // Quadratic spacing: dense near k = 0 where the poles move fastest. Poles
        // also move fastest near a breakaway point (like sqrt(k - k_b)), so land
        // exactly on each one and cluster samples either side of it.
        let mut gains: Vec<f64> = linspace(0.0, 1.0, 800).iter().map(|x| k_max * x * x).collect();
        for (_, kb) in self.breakaway_points().into_iter().filter(|(_, k)| *k <= k_max) {
            gains.push(kb);
            for d in logspace(1e-8, 0.05, 60) {
                gains.extend([kb * (1.0 - d), kb * (1.0 + d)].into_iter().filter(|k| *k <= k_max));
            }
        }
        gains.sort_by(f64::total_cmp);
        gains.dedup();
        let mut p = Plot::new("Root locus").labels("Real", "Imaginary").hline(0.0).vline(0.0);
        for branch in self.root_locus(&gains) {
            p = p.line("", re_im(&branch));
        }
        p = p.markers("open-loop poles", re_im(&self.poles()), Marker::Cross);
        let zeros = self.zeros();
        if !zeros.is_empty() {
            p = p.markers("open-loop zeros", re_im(&zeros), Marker::Circle);
        }
        p
    }
}

impl Dtf {
    /// Step response samples as dots.
    pub fn step_plot(&self, samples: usize) -> Plot {
        let (t, y) = self.step_response(samples);
        let pts = t.into_iter().zip(y).collect();
        Plot::new("Discrete step response").labels("Time (s)", "Output").markers("y[k]", pts, Marker::Dot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_axis_under_a_decade_still_has_ticks() {
        let ticks = log_ticks(2f64.log10(), 8f64.log10());
        let labels: Vec<&str> = ticks.iter().map(|t| t.1.as_str()).collect();
        assert_eq!(labels, ["2", "3", "4", "5", "6", "7", "8"]);
    }

    #[test]
    fn ticks_are_round_numbers() {
        let (step, t) = nice_ticks(0.0, 1.0, false);
        assert_eq!(step, 0.2);
        assert_eq!(t.len(), 6);
        assert_eq!(fmt_num(t[1], step), "0.2");
        let (_, decades) = nice_ticks(-2.0, 2.0, true);
        assert_eq!(decades, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn svg_is_well_formed_enough() {
        let g = Tf::new([1.0], [1.0, 1.0, 1.0]);
        let svg = g.step_plot().to_svg();
        assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
        assert!(svg.contains("<polyline"));
        assert!(!svg.contains("NaN") && !svg.contains("inf"));
    }
}
