// PD controller D(s) = K + K_d s on plant G(s) = 2 / (s^2 + 9), unity feedback.
// Run with: cargo run --example pd_steady_state

use controls::{Input, Tf};

fn main() {
    let g = Tf::new([2.0], [1.0, 0.0, 9.0]);
    println!("G(s) =\n{g}\n");

    for (k, kd) in [(0.0, 1.0), (1.0, 1.0), (10.0, 1.0), (100.0, 1.0), (10.0, 0.0), (-10.0, 1.0)] {
        let l = &Tf::new([kd, k], [1.0]) * &g;
        let show = |e: Option<f64>| e.map_or("unstable".to_string(), |e| format!("{e:.4}"));
        println!(
            "K = {k:>5}, K_d = {kd}: step e_ss = {}, ramp e_ss = {}",
            show(l.steady_state_error(Input::Step)),
            show(l.steady_state_error(Input::Ramp)),
        );
    }

    let closed = (&Tf::new([1.0, 10.0], [1.0]) * &g).unity_feedback();
    println!("\nClosed loop with K = 10, K_d = 1:\n{closed}");
    println!("poles: {:?}", closed.poles());
    let (t, y) = closed.step_response(5.0, 0.001);
    for i in (0..t.len()).step_by(1000) {
        println!("  y({:.0}) = {:.4}", t[i], y[i]);
    }
    let (mag, phase) = closed.bode(3.0);
    println!("at ω = 3 rad/s: {mag:.2} dB, {phase:.1}°");
}
