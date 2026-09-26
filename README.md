# controls

A small signals-and-systems toolbox in Rust: transfer functions, stability,
steady-state error, time and frequency response, root locus, discrete-time
systems, and SVG plots. It has one dependency (`num-complex`).

```rust
use controls::{Input, Tf, save_stacked};

let s = Tf::s();
let g = 2.0 / (&s * &s + 9.0);          // plant 2 / (s^2 + 9)
let l = Tf::pid(10.0, 0.0, 1.0) * &g;   // PD controller 10 + s, in series
let closed = l.unity_feedback();

assert!(closed.is_stable());
println!("{closed}");
//    2s + 20
// -------------
// s^2 + 2s + 29

let e = l.steady_state_error(Input::Step);   // Some(0.3103)
let info = closed.step_info().unwrap();      // overshoot, rise/settling time, ...
let pm = l.margins().phase.unwrap();         // 29.5° at 5.655 rad/s

closed.step_plot().save("step.svg")?;
save_stacked(&l.bode_plot(), "bode.svg")?;
```

## Features

| Area | What's there |
|---|---|
| Polynomials (`Poly`) | `+ - *`, evaluation, roots, Routh–Hurwitz test, construct from roots |
| Transfer functions (`Tf`) | `+` parallel, `*` series, `-`, `/`, scalars; `feedback`, `unity_feedback`; poles, zeros, DC gain |
| Builders | `Tf::s()`, `pid`, `lead_lag`, `zpk`, `butterworth` |
| Stability & error | `is_stable`, `system_type`, `steady_state_error` for step / ramp / parabola |
| Time response | `step_response`, `simulate` with any input `u(t)`, `step_info` (rise, settling, peak, overshoot) |
| Frequency response | `bode`, `bode_sweep` (unwrapped phase), `margins` (gain and phase margin) |
| Root locus | `root_locus(&gains)`, grouped into continuous branches |
| Discrete time (`Dtf`) | `c2d` by zero-order hold or Tustin; poles, stability, `simulate`, `step_response` |
| Signals | `fft`, `ifft`, `convolve`, `linspace`, `logspace` |
| Plots | SVG step, Bode, pole-zero, root-locus, and discrete step plots; a general `Plot` builder |

## Using it

The crate isn't published. Add it by path or git:

```toml
[dependencies]
controls = { path = "../controls" }
```

## Examples

```sh
cargo run --example pd_steady_state   # steady-state error of a PD loop across gains
cargo run --example tour              # every feature; writes SVG plots to plots/
cargo test
```

## Conventions

- Polynomial coefficients are listed highest power first: `s^2 + 9` is `[1.0, 0.0, 9.0]`.
- Transfer functions are stored with a monic denominator. Poles and zeros are
  never cancelled, and `==` compares coefficients exactly.
- `steady_state_error` and `margins` treat `self` as the open loop `L(s)` in a
  unity negative-feedback loop.
- Functions that can have no answer return `Option`: `steady_state_error` and
  `step_info` return `None` for an unstable loop, and a margin is `None` when
  there is no crossover.
- Operators take references (`&a * &b`) or owned values (`a * b`).

## Limitations

- Single-input single-output systems only; there is no public state-space type.
- Roots come from Durand–Kerner iteration. A repeated root of multiplicity `m`
  is accurate to roughly `1e-16^(1/m)`, e.g. about `1e-4` for a 4-fold root.
- Improper transfer functions, such as a PID with a derivative term, can be
  analysed and combined but not simulated or discretized with ZOH.
- The phase of a system with poles or zeros exactly on the jω axis jumps by
  180°; `bode_sweep` unwraps that jump in an arbitrary direction.
