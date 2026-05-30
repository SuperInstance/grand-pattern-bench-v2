//! Grand Pattern Bench v2 — Main benchmark runner
//!
//! Tests three conservation fixes against the broken naive baseline.

use grand_pattern_bench_v2::*;

fn main() {
    let configs = all_configs();
    let num_rooms = 10;
    let ticks = 10_000;
    let anomaly_at = Some(5000);
    let tolerance = 0.1;

    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Grand Pattern Bench v2 — Conservation Law Fix Experiment              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Config: {} rooms × {} ticks × {} configurations", num_rooms, ticks, configs.len());
    println!("Anomaly: sudden 10x magnitude spike at tick 5000 (100 ticks)");
    println!();

    let strategies: Vec<&str> = vec!["naive", "normalized", "matched", "ratio"];
    let mut all_results: Vec<RunResult> = Vec::new();

    for strat_name in &strategies {
        println!("━━━ Strategy: {} ━━━", strat_name);

        for config in &configs {
            let result = match *strat_name {
                "naive" => {
                    let mut s = NaiveStrategy;
                    run_simulation(&mut s, config, num_rooms, ticks, anomaly_at, tolerance)
                }
                "normalized" => {
                    let mut s = NormalizedStrategy;
                    run_simulation(&mut s, config, num_rooms, ticks, anomaly_at, tolerance)
                }
                "matched" => {
                    let mut s = MatchedStrategy;
                    run_simulation(&mut s, config, num_rooms, ticks, anomaly_at, tolerance)
                }
                "ratio" => {
                    let mut s = RatioStrategy::new();
                    run_simulation(&mut s, config, num_rooms, ticks, anomaly_at, tolerance)
                }
                _ => continue,
            };

            println!("  {}", format_result(&result));
            all_results.push(result);
        }
        println!();
    }

    // ─── Summary ──────────────────────────────────────────────────────
    println!("━━━ Summary ━━━");
    println!();

    for strat_name in &strategies {
        let results: Vec<&RunResult> = all_results.iter()
            .filter(|r| r.strategy_name == *strat_name)
            .collect();

        if results.is_empty() { continue; }

        let avg_violation: f64 = results.iter().map(|r| r.violation_rate).sum::<f64>() / results.len() as f64;
        let avg_error: f64 = results.iter().map(|r| r.avg_conservation_error).sum::<f64>() / results.len() as f64;
        let max_error: f64 = results.iter().map(|r| r.max_conservation_error).fold(f64::NEG_INFINITY, f64::max);
        let anomaly_detect_rate: f64 = results.iter().filter(|r| r.anomaly_detected).count() as f64 / results.len() as f64;
        let convergence_rate: f64 = results.iter().filter(|r| r.vibe_converged).count() as f64 / results.len() as f64;
        let avg_spike: f64 = results.iter().map(|r| r.surprise_spike_height).sum::<f64>() / results.len() as f64;

        println!("  {:>12} | violation={:.1}% avg_err={:.4} max_err={:.4} anomaly_detect={:.0}% convergence={:.0}% spike={:.2}",
            strat_name,
            avg_violation * 100.0,
            avg_error,
            max_error,
            anomaly_detect_rate * 100.0,
            convergence_rate * 100.0,
            avg_spike,
        );
    }

    println!();
    println!("Done. {} total configuration runs.", all_results.len());
}
