# flux-perception

> Multi-sensor fusion engine with confidence-weighted aggregation for FLUX agents.

## What This Is

`flux-perception` is a Rust crate implementing a **sensor fusion engine** — it takes readings from multiple sensors, each with their own weight, bias, and confidence, and produces a single `FusedSignal` representing the best estimate of reality.

## Role in the FLUX Ecosystem

Agents don't perceive the world through a single lens. `flux-perception` models how FLUX agents fuse noisy, contradictory inputs into coherent decisions:

- **`flux-navigate`** fuses position estimates from multiple sources
- **`flux-compass`** combines heading readings with sensor data
- **`flux-evolve`** uses fused signals as fitness inputs for behavioral evolution
- **`flux-trust`** cross-references perception confidence with trust scores

## Key Features

| Feature | Description |
|---------|-------------|
| **Weighted Fusion** | Combines sensor values using confidence × weight averaging |
| **Bias Calibration** | Per-sensor bias correction via `calibrate()` |
| **Confidence Threshold** | Ignores sensors below a minimum confidence floor |
| **Agreement Metric** | `agreement()` returns 0.0–1.0 indicating how aligned sensors are |
| **Variance Tracking** | Reports spread/confidence of the fused signal |
| **History** | Tracks recent fused signals for trend analysis |

## Quick Start

```rust
use flux_perception::Engine;

let mut engine = Engine::new(0.1); // confidence threshold

// Add sensors with different weights
engine.add_sensor(1, 1.0, 0.0);  // high-weight primary
engine.add_sensor(2, 0.5, 0.0);  // lower-weight secondary

// Feed readings
engine.update(1, 42.0, 0.95, 100);
engine.update(2, 44.0, 0.70, 100);

// Read fused signal
let fused = engine.read();
println!("Value: {:.1}, Confidence: {:.2}", fused.value, fused.confidence);
println!("Agreement: {:.2}", engine.agreement());

// Calibrate a sensor
engine.calibrate(2, -2.0); // correct systematic bias
engine.deactivate(1);       // ignore a faulty sensor
```

## Building & Testing

```bash
cargo build
cargo test
```

## Related Fleet Repos

- [`flux-navigate`](https://github.com/SuperInstance/flux-navigate) — BFS pathfinding on 32×32 grids
- [`flux-compass`](https://github.com/SuperInstance/flux-compass) — Heading/orientation with spring-damper physics
- [`flux-memory`](https://github.com/SuperInstance/flux-memory) — Key-value store for caching sensor state
- [`flux-trust`](https://github.com/SuperInstance/flux-trust) — Bayesian trust scoring
- [`flux-evolve`](https://github.com/SuperInstance/flux-evolve) — Behavioral evolution engine

## License

Part of the [SuperInstance](https://github.com/SuperInstance) FLUX fleet.
