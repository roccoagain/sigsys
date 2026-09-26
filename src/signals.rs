//! Signal utilities: spacing, convolution, and the discrete Fourier transform.

use std::f64::consts::PI;

use num_complex::Complex64;

/// `n` evenly spaced values from `start` to `end` inclusive.
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    match n {
        0 => vec![],
        1 => vec![start],
        _ => (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect(),
    }
}

/// `n` logarithmically spaced values from `start` to `end` inclusive (both > 0).
pub fn logspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    linspace(start.log10(), end.log10(), n).into_iter().map(|e| 10f64.powf(e)).collect()
}

/// Full linear convolution, length `a.len() + b.len() - 1`.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// Discrete Fourier transform. Radix-2 FFT when the length is a power of two,
/// direct O(n²) DFT otherwise.
pub fn fft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    if !n.is_power_of_two() {
        return (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .map(|(j, v)| v * Complex64::from_polar(1.0, -2.0 * PI * (k * j) as f64 / n as f64))
                    .sum()
            })
            .collect();
    }
    let even: Vec<Complex64> = x.iter().step_by(2).copied().collect();
    let odd: Vec<Complex64> = x.iter().skip(1).step_by(2).copied().collect();
    let (even, odd) = (fft(&even), fft(&odd));
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n / 2 {
        let t = Complex64::from_polar(1.0, -2.0 * PI * k as f64 / n as f64) * odd[k];
        out[k] = even[k] + t;
        out[k + n / 2] = even[k] - t;
    }
    out
}

/// Inverse DFT, so that `ifft(&fft(x)) == x`.
pub fn ifft(x: &[Complex64]) -> Vec<Complex64> {
    let conj: Vec<Complex64> = x.iter().map(|v| v.conj()).collect();
    let n = x.len() as f64;
    fft(&conj).iter().map(|v| v.conj() / n).collect()
}

/// FFT of a real signal.
pub fn fft_real(x: &[f64]) -> Vec<Complex64> {
    fft(&x.iter().map(|v| Complex64::new(*v, 0.0)).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacing() {
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let l = logspace(0.01, 100.0, 5);
        for (got, want) in l.iter().zip([0.01, 0.1, 1.0, 10.0, 100.0]) {
            assert!((got - want).abs() < 1e-12 * want);
        }
    }

    #[test]
    fn convolution() {
        assert_eq!(convolve(&[1.0, 1.0], &[1.0, 2.0]), vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn fft_finds_a_tone() {
        let n = 64;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).cos()).collect();
        let spectrum = fft_real(&x);
        assert!((spectrum[5].norm() - n as f64 / 2.0).abs() < 1e-9);
        assert!(spectrum[6].norm() < 1e-9);
    }

    #[test]
    fn inverse_round_trips_any_length() {
        for n in [8, 12] {
            let x: Vec<Complex64> = (0..n).map(|i| Complex64::new(i as f64, (i * i) as f64)).collect();
            for (a, b) in ifft(&fft(&x)).iter().zip(&x) {
                assert!((a - b).norm() < 1e-9);
            }
        }
    }
}
