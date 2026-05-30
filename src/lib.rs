//! Grand Pattern Bench v2 — Conservation Law Fix Experiment
//!
//! Three proposed fixes for the broken conservation law:
//! - Fix 1: Normalized Bookkeeping (unit-length vectors)
//! - Fix 2: Matched-Magnitude Bookkeeping (prediction error/variance)
//! - Fix 3: Ratio Conservation (stable ratio instead of ≈1.0)

use std::f64::consts::PI;

#[cfg(test)]
mod tests;

// ─── Vector Ops ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct VecN {
    pub data: Vec<f64>,
}

impl VecN {
    pub fn zeros(dim: usize) -> Self {
        Self { data: vec![0.0; dim] }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    pub fn magnitude(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-15 {
            return self.clone();
        }
        Self {
            data: self.data.iter().map(|x| x / mag).collect(),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect(),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect(),
        }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self {
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum()
    }
}

// ─── Config ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Config {
    pub dim: usize,
    pub window: usize,
    pub gc_threshold: usize,
    pub label: String,
}

impl Config {
    pub fn new(dim: usize, window: usize, gc_threshold: usize) -> Self {
        Self {
            dim,
            window,
            gc_threshold,
            label: format!("d={} w={} gc={}", dim, window, gc_threshold),
        }
    }
}

pub fn all_configs() -> Vec<Config> {
    let mut configs = Vec::new();
    let dims = [4, 8, 16];
    let windows = [50, 200, 500];
    let gc_thresholds = [100, 500];

    for &d in &dims {
        for &w in &windows {
            for &gc in &gc_thresholds {
                configs.push(Config::new(d, w, gc));
            }
        }
    }
    configs
}

// ─── Database ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Database {
    pub entries: Vec<VecN>,
}

impl Database {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

impl Database {
    pub fn insert(&mut self, v: VecN) {
        self.entries.push(v);
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }

    pub fn z_magnitude(&self) -> f64 {
        self.entries.iter().map(|v| v.magnitude()).sum()
    }

    pub fn gc(&mut self, threshold: usize) {
        if self.entries.len() > threshold {
            let excess = self.entries.len() - threshold;
            self.entries.drain(0..excess);
        }
    }

    pub fn predict(&self) -> VecN {
        if self.entries.is_empty() {
            return VecN::zeros(0);
        }
        let dim = self.entries[0].dim();
        let n = self.entries.len() as f64;
        let mut sum = vec![0.0; dim];
        for entry in &self.entries {
            for (i, &val) in entry.data.iter().enumerate() {
                sum[i] += val;
            }
        }
        VecN {
            data: sum.into_iter().map(|s| s / n).collect(),
        }
    }

    pub fn prediction_variance(&self) -> f64 {
        if self.entries.len() < 2 {
            return 0.0;
        }
        let mean = self.predict();
        let n = self.entries.len() as f64;
        let mut total_var = 0.0;
        for entry in &self.entries {
            let diff = entry.sub(&mean);
            total_var += diff.dot(&diff);
        }
        total_var / n
    }
}

// ─── Room Simulator ───────────────────────────────────────────────────

/// Simulates a "room" — a time-series generator with configurable dimensionality.
/// Produces perception vectors at each tick.
pub struct Room {
    pub dim: usize,
    pub tick: usize,
    pub phase: Vec<f64>,
    pub anomaly_at: Option<usize>,
    pub anomaly_active: bool,
}

impl Room {
    pub fn new(dim: usize, anomaly_at: Option<usize>) -> Self {
        let mut phase = Vec::with_capacity(dim);
        for i in 0..dim {
            phase.push(i as f64 * 2.0 * PI / dim as f64);
        }
        Self {
            dim,
            tick: 0,
            phase,
            anomaly_at,
            anomaly_active: false,
        }
    }

    pub fn perceive(&mut self) -> VecN {
        let t = self.tick as f64;
        let mut data = Vec::with_capacity(self.dim);

        for (i, &p) in self.phase.iter().enumerate() {
            // Multi-frequency signal with drift
            let freq1 = 0.01 + 0.002 * (i as f64);
            let freq2 = 0.037;
            let base = (t * freq1 + p).sin() * 3.0 + (t * freq2 + p * 0.5).cos() * 1.5;
            let drift = t * 0.001; // Linear drift — the conservation killer
            data.push(base + drift);
        }

        // Inject anomaly if requested
        if let Some(at) = self.anomaly_at {
            if self.tick >= at && self.tick < at + 100 {
                self.anomaly_active = true;
                for d in &mut data {
                    *d *= 10.0;
                }
            } else {
                self.anomaly_active = false;
            }
        }

        self.tick += 1;
        VecN { data }
    }
}

// ─── Strategies ───────────────────────────────────────────────────────

/// How a strategy stores perception/prediction into db_in and db_out
pub trait Strategy {
    fn name(&self) -> &str;
    fn store_in(&self, perception: &VecN, prediction: &VecN, db_in: &mut Database);
    fn store_out(&self, prediction: &VecN, db_out: &mut Database);
    fn conservation_error(&self, db_in: &Database, db_out: &Database) -> f64;
    fn is_conserved(&self, db_in: &Database, db_out: &Database, tolerance: f64) -> bool;
}

/// Naive baseline — the broken approach
pub struct NaiveStrategy;

impl Strategy for NaiveStrategy {
    fn name(&self) -> &str { "naive" }

    fn store_in(&self, perception: &VecN, _prediction: &VecN, db_in: &mut Database) {
        db_in.insert(perception.clone());
    }

    fn store_out(&self, prediction: &VecN, db_out: &mut Database) {
        db_out.insert(prediction.clone());
    }

    fn conservation_error(&self, db_in: &Database, db_out: &Database) -> f64 {
        let z_in = db_in.z_magnitude();
        let z_out = db_out.z_magnitude();
        if z_in + z_out < 1e-15 { return 0.0; }
        (z_in - z_out).abs() / ((z_in + z_out) / 2.0)
    }

    fn is_conserved(&self, db_in: &Database, db_out: &Database, tolerance: f64) -> bool {
        self.conservation_error(db_in, db_out) < tolerance
    }
}

/// Fix 1: Normalized Bookkeeping
pub struct NormalizedStrategy;

impl Strategy for NormalizedStrategy {
    fn name(&self) -> &str { "normalized" }

    fn store_in(&self, perception: &VecN, _prediction: &VecN, db_in: &mut Database) {
        db_in.insert(perception.normalized());
    }

    fn store_out(&self, prediction: &VecN, db_out: &mut Database) {
        db_out.insert(prediction.normalized());
    }

    fn conservation_error(&self, db_in: &Database, db_out: &Database) -> f64 {
        let z_in = db_in.z_magnitude();
        let z_out = db_out.z_magnitude();
        if z_in + z_out < 1e-15 { return 0.0; }
        // With normalization, |Z_in| ≈ count_in, |Z_out| ≈ count_out
        (z_in - z_out).abs() / ((z_in + z_out) / 2.0)
    }

    fn is_conserved(&self, db_in: &Database, db_out: &Database, tolerance: f64) -> bool {
        self.conservation_error(db_in, db_out) < tolerance
    }
}

/// Fix 2: Matched-Magnitude Bookkeeping
/// Store prediction error (perception - prediction) normalized by prediction magnitude
/// in db_in, and unit prediction direction in db_out.
/// Both are dimensionless, magnitude-matched quantities.
pub struct MatchedStrategy;

impl Strategy for MatchedStrategy {
    fn name(&self) -> &str { "matched" }

    fn store_in(&self, perception: &VecN, prediction: &VecN, db_in: &mut Database) {
        // Store normalized prediction error: (perception - prediction) / ||perception||
        // This is dimensionless and bounded
        let error = perception.sub(prediction);
        let p_mag = perception.magnitude();
        if p_mag > 1e-10 {
            db_in.insert(error.scale(1.0 / p_mag));
        } else {
            db_in.insert(error);
        }
    }

    fn store_out(&self, prediction: &VecN, db_out: &mut Database) {
        // Store normalized prediction direction
        db_out.insert(prediction.normalized());
    }

    fn conservation_error(&self, db_in: &Database, db_out: &Database) -> f64 {
        let z_in = db_in.z_magnitude();
        let z_out = db_out.z_magnitude();
        if z_in + z_out < 1e-15 { return 0.0; }
        (z_in - z_out).abs() / ((z_in + z_out) / 2.0)
    }

    fn is_conserved(&self, db_in: &Database, db_out: &Database, tolerance: f64) -> bool {
        self.conservation_error(db_in, db_out) < tolerance
    }
}

/// Fix 3: Ratio Conservation
pub struct RatioStrategy {
    pub history: Vec<f64>,
}

impl RatioStrategy {
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }
}

impl Default for RatioStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl RatioStrategy {
    pub fn ratio_stability(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean: f64 = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let variance: f64 = self.history.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / self.history.len() as f64;
        variance.sqrt() // std dev of ratio
    }

    pub fn is_ratio_stable(&self, max_stddev: f64) -> bool {
        self.ratio_stability() < max_stddev
    }
}

impl Strategy for RatioStrategy {
    fn name(&self) -> &str { "ratio" }

    fn store_in(&self, perception: &VecN, _prediction: &VecN, db_in: &mut Database) {
        db_in.insert(perception.clone());
    }

    fn store_out(&self, prediction: &VecN, db_out: &mut Database) {
        db_out.insert(prediction.clone());
    }

    fn conservation_error(&self, db_in: &Database, db_out: &Database) -> f64 {
        let z_in = db_in.z_magnitude();
        let z_out = db_out.z_magnitude();
        if z_out < 1e-15 { return 0.0; }
        let ratio = z_in / z_out;
        // We don't check error here — ratio stability is checked separately
        ratio
    }

    fn is_conserved(&self, _db_in: &Database, _db_out: &Database, _tolerance: f64) -> bool {
        // For ratio strategy, conservation is about stability, not ≈1.0
        // This is checked externally via ratio_stability()
        true
    }
}

// ─── Simulation Runner ───────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RunResult {
    pub config_label: String,
    pub strategy_name: String,
    pub max_conservation_error: f64,
    pub avg_conservation_error: f64,
    pub violation_rate: f64,
    pub final_vibe: f64,
    pub vibe_converged: bool,
    pub anomaly_detected: bool,
    pub surprise_spike_height: f64,
    pub ratio_stddev: f64,
}

pub fn run_simulation(
    strategy: &mut dyn Strategy,
    config: &Config,
    num_rooms: usize,
    ticks: usize,
    anomaly_at: Option<usize>,
    tolerance: f64,
) -> RunResult {
    run_simulation_opts(strategy, config, num_rooms, ticks, anomaly_at, tolerance, false)
}

pub fn run_simulation_opts(
    strategy: &mut dyn Strategy,
    config: &Config,
    num_rooms: usize,
    ticks: usize,
    anomaly_at: Option<usize>,
    tolerance: f64,
    track_ratio: bool,
) -> RunResult {
    let mut total_error = 0.0;
    let mut max_error = 0.0;
    let mut violations = 0usize;
    let mut total_checks = 0usize;
    let mut final_vibes = Vec::new();
    let mut max_surprise = 0.0;
    let mut anomaly_detected = false;
    let mut ratio_history: Vec<f64> = Vec::new();

    for room_id in 0..num_rooms {
        let mut room = Room::new(config.dim, anomaly_at);
        let mut db_in = Database::new();
        let mut db_out = Database::new();
        let mut prev_surprise = 0.0f64;

        for tick in 0..ticks {
            let perception = room.perceive();

            // Bootstrap: first few ticks just fill the databases
            if tick < config.window {
                strategy.store_in(&perception, &VecN::zeros(config.dim), &mut db_in);
                // No prediction yet for out
            } else {
                let prediction = db_in.predict();

                // Compute surprise: ||perception - prediction||
                let diff = perception.sub(&prediction);
                let surprise = diff.magnitude();

                // Check for anomaly spike
                if anomaly_at.is_some_and(|at| tick >= at && tick < at + 100) {
                    if surprise > max_surprise {
                        max_surprise = surprise;
                    }
                    // Anomaly detected if surprise jumps significantly
                    if prev_surprise > 0.0 && surprise > prev_surprise * 3.0 {
                        anomaly_detected = true;
                    }
                }
                prev_surprise = surprise;

                strategy.store_in(&perception, &prediction, &mut db_in);
                strategy.store_out(&prediction, &mut db_out);

                // Periodic conservation check
                if tick % 100 == 0 && db_in.count() > 0 && db_out.count() > 0 {
                    let err = strategy.conservation_error(&db_in, &db_out);
                    total_error += err;
                    total_checks += 1;
                    if err > max_error { max_error = err; }
                    if !strategy.is_conserved(&db_in, &db_out, tolerance) {
                        violations += 1;
                    }

                    // Track ratio for Fix 3
                    if track_ratio {
                        ratio_history.push(err);
                    }
                }

                db_in.gc(config.gc_threshold);
                db_out.gc(config.gc_threshold);
            }
        }

        // Final vibe: how well prediction matches recent perceptions
        if db_in.count() > 0 {
            let pred = db_in.predict();
            let mag_pred = pred.magnitude();
            final_vibes.push(mag_pred);
        }

        let _ = room_id;
    }

    let avg_error = if total_checks > 0 { total_error / total_checks as f64 } else { 0.0 };
    let violation_rate = if total_checks > 0 { violations as f64 / total_checks as f64 } else { 0.0 };
    let avg_vibe: f64 = if !final_vibes.is_empty() {
        final_vibes.iter().sum::<f64>() / final_vibes.len() as f64
    } else {
        0.0
    };

    // Vibe converged if spread is small
    let vibe_spread = if final_vibes.len() > 1 {
        let mean = avg_vibe;
        let var: f64 = final_vibes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / final_vibes.len() as f64;
        var.sqrt()
    } else {
        0.0
    };

    let ratio_stddev = if ratio_history.len() > 1 {
        let mean: f64 = ratio_history.iter().sum::<f64>() / ratio_history.len() as f64;
        let var: f64 = ratio_history.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratio_history.len() as f64;
        var.sqrt()
    } else {
        0.0
    };

    RunResult {
        config_label: config.label.clone(),
        strategy_name: strategy.name().to_string(),
        max_conservation_error: max_error,
        avg_conservation_error: avg_error,
        violation_rate,
        final_vibe: avg_vibe,
        vibe_converged: vibe_spread < 1.0,
        anomaly_detected,
        surprise_spike_height: max_surprise,
        ratio_stddev,
    }
}

// ─── Report ───────────────────────────────────────────────────────────

pub fn format_result(r: &RunResult) -> String {
    format!(
        "{:<20} | err_max={:.4} err_avg={:.4} viol={:.1}pct vibe={:.2} converged={} anomaly={} spike={:.2} ratio_std={:.4}",
        format!("[{}] {}", r.strategy_name, r.config_label),
        r.max_conservation_error,
        r.avg_conservation_error,
        r.violation_rate * 100.0,
        r.final_vibe,
        r.vibe_converged,
        r.anomaly_detected,
        r.surprise_spike_height,
        r.ratio_stddev,
    )
}
