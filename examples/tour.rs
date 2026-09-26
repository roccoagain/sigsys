// A walk through the toolbox on the PD-controlled plant 2 / (s^2 + 9).
// Run with: cargo run --example tour   (plots are written to ./plots/)

use sigsys::signals::{convolve, fft_real};
use sigsys::{Discretize, Tf, save_stacked};

fn main() -> std::io::Result<()> {
    std::fs::create_dir_all("plots")?;
    let s = Tf::s();

    // 5. Build systems algebraically.
    let g = 2.0 / (&s * &s + 9.0);
    let d = Tf::pid(10.0, 0.0, 1.0); // K = 10, K_d = 1
    let l = &d * &g;
    let closed = l.unity_feedback();
    println!("Closed loop:\n{closed}\n");

    // 3. Plots.
    closed.step_plot().save("plots/step.svg")?;
    save_stacked(&l.bode_plot(), "plots/bode.svg")?;
    closed.pzmap_plot().save("plots/pzmap.svg")?;
    (2.0 * (&s + 10.0) / (&s * &s + 9.0)).root_locus_plot(40.0).save("plots/rlocus.svg")?;
    println!("Wrote plots/step.svg, bode.svg, pzmap.svg, rlocus.svg\n");

    // 1. Step metrics.
    let info = closed.step_info().expect("stable");
    println!(
        "Step: rise {:.3} s, settling {:.3} s, peak {:.4} at {:.3} s, overshoot {:.1}%, final {:.4}\n",
        info.rise_time, info.settling_time, info.peak, info.peak_time, info.overshoot_pct, info.final_value
    );

    // 2. Margins of the open loop.
    let m = l.margins();
    match m.phase {
        Some(c) => println!("Phase margin: {:.1}° at {:.3} rad/s", c.margin, c.w),
        None => println!("Phase margin: infinite (no 0 dB crossing)"),
    }
    match m.gain_margin_db() {
        Some(db) => println!("Gain margin: {db:.1} dB"),
        None => println!("Gain margin: infinite (phase never reaches -180°)\n"),
    }

    // 6. Discrete time.
    let dz = closed.c2d(0.05, Discretize::Zoh);
    println!("ZOH, T = 0.05:\n{dz}\nstable: {}, DC gain {:.4}\n", dz.is_stable(), dz.dc_gain());
    dz.step_plot(100).save("plots/discrete_step.svg")?;

    // 7. Signals.
    let lp = Tf::butterworth(4, 2.0);
    println!("4th-order Butterworth, ωc = 2:\n{lp}\n");
    save_stacked(&lp.bode_plot(), "plots/butterworth.svg")?;
    println!("[1,1] * [1,2] = {:?}", convolve(&[1.0, 1.0], &[1.0, 2.0]));
    let x: Vec<f64> = (0..16).map(|i| (std::f64::consts::PI * i as f64 / 4.0).sin()).collect();
    let peak = fft_real(&x).iter().take(8).enumerate().max_by(|a, b| a.1.norm().total_cmp(&b.1.norm())).unwrap().0;
    println!("FFT of a 2-cycles-per-16-samples sine peaks at bin {peak}");
    Ok(())
}
