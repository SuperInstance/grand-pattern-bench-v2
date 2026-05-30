# Grand Pattern Bench v2 — Fixing the Conservation Law

> **Root Cause:** Raw perceptions have higher magnitude than averaged predictions, causing linear drift. The benchmark proved it: 100% violation rate across 1.8M room-tick pairs.

## Three Proposed Fixes

### Fix 1: Normalized Bookkeeping
Before inserting into `db_in` or `db_out`, normalize vectors to unit length: `v = v / ||v||`
This ensures `|Z_in| = count_in` and `|Z_out| = count_out` (always comparable).

### Fix 2: Matched-Magnitude Bookkeeping
Store the prediction error (`perception - prediction`) in `db_in` and the prediction itself in `db_out`.
Both have naturally matched magnitudes because they're derived from the same scale.

### Fix 3: Ratio Conservation
Instead of `|Z_in| ≈ |Z_out|`, check that the ratio `|Z_in|/|Z_out|` stays within bounds.
Conservation means the ratio is **stable**, not necessarily 1.0.

## Experiment Design

- **18 configurations:** 3 dimensions (4, 8, 16) × 3 windows (50, 200, 500) × 2 GC thresholds (100, 500)
- **10 rooms**, **10,000 ticks** each
- **Anomaly injection** at tick 5000: sudden 10× magnitude spike (100 ticks)
- **Metrics:** max error, avg error, violation rate, vibe convergence, surprise spike height

## Results

Run the benchmark:
```bash
cargo run --release
```

Run tests (20+ tests):
```bash
cargo test
```

## Architecture

- **Zero dependencies** — pure Rust, no external crates
- `lib.rs` — Core simulation engine: strategies, database, room simulator, runner
- `main.rs` — Benchmark runner across all 4 strategies × 18 configs
- `tests.rs` — 20 comprehensive tests covering all fixes, edge cases, and properties

## Key Insight

The naive approach stores raw perceptions (growing magnitude from drift) alongside averaged predictions (smoothed magnitude). This fundamental scale mismatch guarantees conservation violation regardless of parameters. All three fixes address this by either normalizing away the scale (Fix 1), working in the same scale (Fix 2), or accepting the scale difference and measuring stability instead (Fix 3).
