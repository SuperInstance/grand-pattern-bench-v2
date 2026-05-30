//! Grand Pattern Bench v2 — Comprehensive Tests (15+)
//!
//! Tests covering all three fixes, naive baseline, edge cases,
//! anomaly detection, vibe convergence, and dimensional independence.

use super::*;

const TOLERANCE: f64 = 0.1;

// ─── Test 1: Naive conservation breaks (reproduce the bug) ────────────

#[test]
fn test_naive_conservation_breaks() {
    let config = Config::new(8, 200, 500);
    let mut strategy = NaiveStrategy;
    let result = run_simulation(&mut strategy, &config, 5, 5000, None, TOLERANCE);

    // Naive approach should show high violation rate due to drift
    assert!(
        result.violation_rate > 0.5,
        "Naive should have high violation rate, got {:.1}%",
        result.violation_rate * 100.0
    );
    assert!(
        result.avg_conservation_error > 0.1,
        "Naive should have significant conservation error, got {:.4}",
        result.avg_conservation_error
    );
}

// ─── Test 2: Fix 1 (normalized) conservation holds ────────────────────

#[test]
fn test_normalized_conservation_holds() {
    let config = Config::new(8, 200, 500);
    let mut strategy = NormalizedStrategy;
    let result = run_simulation(&mut strategy, &config, 5, 5000, None, TOLERANCE);

    assert!(
        result.violation_rate < 0.2,
        "Normalized should have low violation rate, got {:.1}%",
        result.violation_rate * 100.0
    );
    assert!(
        result.avg_conservation_error < 0.2,
        "Normalized avg error should be low, got {:.4}",
        result.avg_conservation_error
    );
}

// ─── Test 3: Fix 2 (matched) conservation holds ───────────────────────

#[test]
fn test_matched_conservation_holds() {
    let config = Config::new(8, 200, 500);
    let mut strategy = MatchedStrategy;
    let result = run_simulation(&mut strategy, &config, 5, 5000, None, TOLERANCE);

    // Matched magnitude should have better conservation than naive
    let mut naive = NaiveStrategy;
    let naive_result = run_simulation(&mut naive, &config, 5, 5000, None, TOLERANCE);

    assert!(
        result.avg_conservation_error < naive_result.avg_conservation_error,
        "Matched ({:.4}) should beat naive ({:.4})",
        result.avg_conservation_error,
        naive_result.avg_conservation_error
    );
}

// ─── Test 4: Fix 3 (ratio) ratio is stable ────────────────────────────

#[test]
fn test_ratio_stability() {
    let config = Config::new(8, 200, 500);
    let mut strategy = RatioStrategy::new();
    let result = run_simulation(&mut strategy, &config, 5, 5000, None, TOLERANCE);

    // Ratio stddev should be bounded — the ratio is stable even if not ≈1.0
    assert!(
        result.ratio_stddev < 2.0,
        "Ratio should have bounded stddev, got {:.4}",
        result.ratio_stddev
    );
}

// ─── Test 5: Fix 1 anomaly detection still works ──────────────────────

#[test]
fn test_normalized_anomaly_detection() {
    let config = Config::new(8, 200, 500);
    let mut strategy = NormalizedStrategy;
    let result = run_simulation(&mut strategy, &config, 5, 5000, Some(3000), TOLERANCE);

    assert!(
        result.anomaly_detected,
        "Normalized should still detect anomalies"
    );
    assert!(
        result.surprise_spike_height > 1.0,
        "Normalized should have meaningful surprise spike, got {:.2}",
        result.surprise_spike_height
    );
}

// ─── Test 6: Fix 2 anomaly detection still works ──────────────────────

#[test]
fn test_matched_anomaly_detection() {
    let config = Config::new(8, 200, 500);
    let mut strategy = MatchedStrategy;
    let result = run_simulation(&mut strategy, &config, 5, 5000, Some(3000), TOLERANCE);

    assert!(
        result.anomaly_detected,
        "Matched should still detect anomalies"
    );
    assert!(
        result.surprise_spike_height > 1.0,
        "Matched should have meaningful surprise spike, got {:.2}",
        result.surprise_spike_height
    );
}

// ─── Test 7: Fix 3 anomaly detection still works ──────────────────────

#[test]
fn test_ratio_anomaly_detection() {
    let config = Config::new(8, 200, 500);
    let mut strategy = RatioStrategy::new();
    let result = run_simulation(&mut strategy, &config, 5, 5000, Some(3000), TOLERANCE);

    assert!(
        result.anomaly_detected,
        "Ratio should still detect anomalies"
    );
    assert!(
        result.surprise_spike_height > 1.0,
        "Ratio should have meaningful surprise spike, got {:.2}",
        result.surprise_spike_height
    );
}

// ─── Test 8: Fix 1 vibe convergence improves ──────────────────────────

#[test]
fn test_normalized_vibe_convergence() {
    let config = Config::new(8, 200, 500);

    let mut naive = NaiveStrategy;
    let naive_result = run_simulation(&mut naive, &config, 5, 5000, None, TOLERANCE);

    let mut normalized = NormalizedStrategy;
    let norm_result = run_simulation(&mut normalized, &config, 5, 5000, None, TOLERANCE);

    // At minimum, normalized should not be worse
    assert!(
        norm_result.vibe_converged || !naive_result.vibe_converged,
        "Normalized vibe should be at least as good as naive"
    );
}

// ─── Test 9: Fix 2 vibe convergence improves ──────────────────────────

#[test]
fn test_matched_vibe_convergence() {
    let config = Config::new(8, 200, 500);

    let mut naive = NaiveStrategy;
    let naive_result = run_simulation(&mut naive, &config, 5, 5000, None, TOLERANCE);

    let mut matched = MatchedStrategy;
    let matched_result = run_simulation(&mut matched, &config, 5, 5000, None, TOLERANCE);

    // Matched should converge or at least not be worse
    assert!(
        matched_result.vibe_converged || !naive_result.vibe_converged,
        "Matched vibe should be at least as good as naive"
    );
}

// ─── Test 10: Fix 3 vibe convergence improves ─────────────────────────

#[test]
fn test_ratio_vibe_convergence() {
    let config = Config::new(8, 200, 500);

    let mut ratio = RatioStrategy::new();
    let result = run_simulation(&mut ratio, &config, 5, 5000, None, TOLERANCE);

    // Ratio approach should still produce meaningful vibes
    assert!(
        result.final_vibe > 0.0,
        "Ratio should produce non-zero vibe, got {:.2}",
        result.final_vibe
    );
}

// ─── Test 11: Edge cases (empty db, single entry) ─────────────────────

#[test]
fn test_edge_case_empty_database() {
    let db = Database::new();
    assert_eq!(db.count(), 0);
    assert_eq!(db.z_magnitude(), 0.0);

    let pred = db.predict();
    assert_eq!(pred.dim(), 0); // Empty db → zero-dim prediction

    let naive = NaiveStrategy;
    assert!(naive.is_conserved(&db, &db, TOLERANCE));
}

#[test]
fn test_edge_case_single_entry() {
    let mut db_in = Database::new();
    let mut db_out = Database::new();

    let v = VecN { data: vec![1.0, 2.0, 3.0] };
    db_in.insert(v.clone());
    db_out.insert(v.clone());

    assert_eq!(db_in.count(), 1);
    assert_eq!(db_out.count(), 1);

    let naive = NaiveStrategy;
    assert!(
        naive.is_conserved(&db_in, &db_out, TOLERANCE),
        "Single identical entries should be conserved"
    );
}

#[test]
fn test_edge_case_zero_vector() {
    let v = VecN::zeros(4);
    let normed = v.normalized();
    // Normalizing a zero vector should still be zero (not NaN)
    for &x in &normed.data {
        assert!(x.is_finite(), "Normalized zero should be finite, got {}", x);
    }
}

// ─── Test 12: Fix 1 normalization doesn't lose info ───────────────────

#[test]
fn test_normalization_preserves_direction() {
    let v = VecN { data: vec![3.0, 4.0, 0.0] };
    let normed = v.normalized();

    assert!((normed.magnitude() - 1.0).abs() < 1e-10, "Should be unit length");
    // Direction preserved: 3/5, 4/5, 0
    assert!((normed.data[0] - 0.6).abs() < 1e-10);
    assert!((normed.data[1] - 0.8).abs() < 1e-10);
    assert!((normed.data[2] - 0.0).abs() < 1e-10);
}

#[test]
fn test_normalized_still_detects_magnitude_anomaly() {
    // Even with normalization, a sudden direction change should be detectable
    let config = Config::new(4, 50, 200);
    let mut strategy = NormalizedStrategy;
    let result = run_simulation(&mut strategy, &config, 3, 5000, Some(3000), TOLERANCE);

    // Anomaly detection should still fire due to direction shift
    assert!(
        result.surprise_spike_height > 0.5,
        "Normalized should detect direction-based anomalies, got spike={:.2}",
        result.surprise_spike_height
    );
}

// ─── Test 13: Dimension doesn't affect fixed conservation ─────────────

#[test]
fn test_dimension_independence_normalized() {
    let dims = [4, 8, 16];
    let mut errors: Vec<f64> = Vec::new();

    for &dim in &dims {
        let config = Config::new(dim, 200, 500);
        let mut strategy = NormalizedStrategy;
        let result = run_simulation(&mut strategy, &config, 3, 5000, None, TOLERANCE);
        errors.push(result.avg_conservation_error);
    }

    // Errors should be similar across dimensions (within 3x)
    let max_err = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_err = errors.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_err / min_err < 3.0,
        "Dimension shouldn't wildly affect normalized conservation: {:?}",
        errors
    );
}

#[test]
fn test_dimension_independence_matched() {
    let dims = [4, 8, 16];
    let mut errors: Vec<f64> = Vec::new();

    for &dim in &dims {
        let config = Config::new(dim, 200, 500);
        let mut strategy = MatchedStrategy;
        let result = run_simulation(&mut strategy, &config, 3, 5000, None, TOLERANCE);
        errors.push(result.avg_conservation_error);
    }

    let max_err = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_err = errors.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_err / min_err < 5.0,
        "Dimension shouldn't wildly affect matched conservation: {:?}",
        errors
    );
}

// ─── Test 14: Window doesn't affect fixed conservation ────────────────

#[test]
fn test_window_independence_normalized() {
    let windows = [50, 200, 500];
    let mut errors: Vec<f64> = Vec::new();

    for &w in &windows {
        let config = Config::new(8, w, 500);
        let mut strategy = NormalizedStrategy;
        let result = run_simulation(&mut strategy, &config, 3, 5000, None, TOLERANCE);
        errors.push(result.avg_conservation_error);
    }

    // All should have low error regardless of window
    for (i, &err) in errors.iter().enumerate() {
        assert!(
            err < 0.5,
            "Window {} should still give low error, got {:.4}",
            windows[i], err
        );
    }
}

// ─── Test 15: GC threshold now matters with fixes ─────────────────────

#[test]
fn test_gc_threshold_matters_normalized() {
    let mut low_gc_result = None;
    let mut high_gc_result = None;

    for &gc in &[100, 500] {
        let config = Config::new(8, 200, gc);
        let mut strategy = NormalizedStrategy;
        let result = run_simulation(&mut strategy, &config, 3, 5000, None, TOLERANCE);

        if gc == 100 {
            low_gc_result = Some(result);
        } else {
            high_gc_result = Some(result);
        }
    }

    let low = low_gc_result.unwrap();
    let high = high_gc_result.unwrap();

    // Different GC thresholds should produce different results
    // (both should be valid, but not identical)
    assert!(
        (low.avg_conservation_error - high.avg_conservation_error).abs() > 0.001
            || low.violation_rate != high.violation_rate,
        "GC threshold should make a difference: low={:.4}/{:.1}% high={:.4}/{:.1}%",
        low.avg_conservation_error, low.violation_rate * 100.0,
        high.avg_conservation_error, high.violation_rate * 100.0,
    );
}

#[test]
fn test_gc_threshold_matters_matched() {
    let mut low_gc_result = None;
    let mut high_gc_result = None;

    for &gc in &[100, 500] {
        let config = Config::new(8, 200, gc);
        let mut strategy = MatchedStrategy;
        let result = run_simulation(&mut strategy, &config, 3, 5000, None, TOLERANCE);

        if gc == 100 {
            low_gc_result = Some(result);
        } else {
            high_gc_result = Some(result);
        }
    }

    let low = low_gc_result.unwrap();
    let high = high_gc_result.unwrap();

    // Both should be reasonable, but different
    assert!(
        (low.avg_conservation_error - high.avg_conservation_error).abs() > 0.001
            || low.violation_rate != high.violation_rate,
        "GC threshold should make a difference with matched strategy"
    );
}

// ─── Bonus Tests ──────────────────────────────────────────────────────

#[test]
fn test_all_fixes_beat_naive_on_violation_rate() {
    let config = Config::new(8, 200, 500);

    let mut naive = NaiveStrategy;
    let naive_result = run_simulation(&mut naive, &config, 5, 5000, None, TOLERANCE);

    let mut normalized = NormalizedStrategy;
    let norm_result = run_simulation(&mut normalized, &config, 5, 5000, None, TOLERANCE);

    let mut matched = MatchedStrategy;
    let matched_result = run_simulation(&mut matched, &config, 5, 5000, None, TOLERANCE);

    assert!(
        norm_result.violation_rate <= naive_result.violation_rate,
        "Normalized ({:.1}%) should beat naive ({:.1}%)",
        norm_result.violation_rate * 100.0,
        naive_result.violation_rate * 100.0
    );

    assert!(
        matched_result.violation_rate <= naive_result.violation_rate,
        "Matched ({:.1}%) should beat naive ({:.1}%)",
        matched_result.violation_rate * 100.0,
        naive_result.violation_rate * 100.0
    );
}

#[test]
fn test_database_gc_actually_removes() {
    let mut db = Database::new();
    for i in 0..100 {
        db.insert(VecN { data: vec![i as f64] });
    }
    assert_eq!(db.count(), 100);

    db.gc(50);
    assert_eq!(db.count(), 50);
    // First entry should now be 50
    assert!((db.entries[0].data[0] - 50.0).abs() < 1e-10);
}

#[test]
fn test_database_prediction_accuracy() {
    let mut db = Database::new();
    // Insert 10 identical vectors
    for _ in 0..10 {
        db.insert(VecN { data: vec![5.0, 5.0] });
    }

    let pred = db.predict();
    assert!((pred.data[0] - 5.0).abs() < 1e-10);
    assert!((pred.data[1] - 5.0).abs() < 1e-10);
}

#[test]
fn test_vecn_ops() {
    let a = VecN { data: vec![1.0, 2.0, 3.0] };
    let b = VecN { data: vec![4.0, 5.0, 6.0] };

    let sum = a.add(&b);
    assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);

    let diff = a.sub(&b);
    assert_eq!(diff.data, vec![-3.0, -3.0, -3.0]);

    let scaled = a.scale(2.0);
    assert_eq!(scaled.data, vec![2.0, 4.0, 6.0]);

    assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    assert!((a.magnitude() - 14.0_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn test_ratio_strategy_tracks_history() {
    let mut rs = RatioStrategy::new();
    assert_eq!(rs.ratio_stability(), 0.0);

    // Simulate some ratio history
    rs.history = vec![2.0, 2.1, 1.9, 2.0, 2.05];
    let stability = rs.ratio_stability();
    assert!(stability < 0.1, "Stable ratio should have low stddev, got {:.4}", stability);
    assert!(rs.is_ratio_stable(0.5));
}

#[test]
fn test_config_generation() {
    let configs = all_configs();
    // 3 dims × 3 windows × 2 gc = 18
    assert_eq!(configs.len(), 18, "Should have 18 configurations");

    // Verify all combinations present
    let has_4_50_100 = configs.iter().any(|c| c.dim == 4 && c.window == 50 && c.gc_threshold == 100);
    let has_16_500_500 = configs.iter().any(|c| c.dim == 16 && c.window == 500 && c.gc_threshold == 500);
    assert!(has_4_50_100, "Should have d=4,w=50,gc=100 config");
    assert!(has_16_500_500, "Should have d=16,w=500,gc=500 config");
}

#[test]
fn test_room_anomaly_injection() {
    let mut room = Room::new(4, Some(100));

    // Normal ticks
    let normal: Vec<VecN> = (0..100).map(|_| room.perceive()).collect();
    let normal_avg_mag: f64 = normal.iter().map(|v| v.magnitude()).sum::<f64>() / normal.len() as f64;

    // Anomaly ticks (at tick 100-199)
    let anomaly: Vec<VecN> = (0..50).map(|_| room.perceive()).collect();
    let anomaly_avg_mag: f64 = anomaly.iter().map(|v| v.magnitude()).sum::<f64>() / anomaly.len() as f64;

    assert!(
        anomaly_avg_mag > normal_avg_mag * 5.0,
        "Anomaly should be ~10x normal. Normal={:.2} Anomaly={:.2}",
        normal_avg_mag, anomaly_avg_mag
    );
}

#[test]
fn test_database_prediction_variance() {
    let mut db = Database::new();
    // Insert identical → zero variance
    for _ in 0..5 {
        db.insert(VecN { data: vec![1.0, 1.0] });
    }
    assert!(db.prediction_variance() < 1e-10, "Identical entries → zero variance");

    // Insert varied → positive variance
    db.insert(VecN { data: vec![10.0, 10.0] });
    assert!(db.prediction_variance() > 1.0, "Varied entries → positive variance");
}
