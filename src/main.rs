// Unity feedback, step input.
// L(s) = D(s)G(s) = 2(K + K_d s) / (s^2 + 9)
// Closed-loop char. eq: s^2 + 2K_d s + (9 + 2K) = 0
// e_ss = 1 / (1 + L(0)) = 9 / (9 + 2K), valid only if the closed loop is stable.

fn steady_state_error(k: f64, kd: f64) -> Option<f64> {
    let stable = kd > 0.0 && 9.0 + 2.0 * k > 0.0; // 2nd order: all coefficients positive
    if !stable {
        return None;
    }
    let kp = 2.0 * k / 9.0; // position constant L(0)
    Some(1.0 / (1.0 + kp))
}

fn main() {
    let kd = 1.0;
    for k in [0.0, 1.0, 4.5, 10.0, 100.0] {
        match steady_state_error(k, kd) {
            Some(e) => println!("K = {k:>5}, K_d = {kd}: e_ss = {e:.4}"),
            None => println!("K = {k:>5}, K_d = {kd}: unstable"),
        }
    }
}
