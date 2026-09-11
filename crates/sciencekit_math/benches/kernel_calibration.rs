//! Grain calibration protocol for the dense kernels (spec `higher-order-kernels`).
//!
//! Measures each kernel across closure costs (cheap/medium/expensive) and sizes
//! 10¹–10⁷, for sequential (`parallelism = 1`) and parallel (`parallelism` =
//! available cores) plans. The smallest size at which the parallel plan beats
//! the sequential one is the kernel's crossover, from which the documented
//! grain (minimum elements per thread) is derived.
//!
//! Run with `cargo bench -p sciencekit_math --bench kernel_calibration` and keep
//! the raw report under `temporary/YYYY-MM-DD/parallel-kernel-execution/`. The
//! resulting constants and their provenance live in
//! `crates/sciencekit_math/src/kernels/grain.rs`.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use sciencekit_math::{
    sk_axis_sum, sk_binary_combine, sk_elementwise_transform, sk_scale_in_place,
};

/// Calibration sizes: 10¹ through 10⁷ elements.
const SIZES: [usize; 7] = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];

/// Cheap closure: one add per element.
fn cheap(value: f64) -> f64 {
    value + 1.0
}

/// Medium closure: one transcendental per element.
fn medium(value: f64) -> f64 {
    value.sin()
}

/// Expensive closure: sixteen transcendental rounds per element.
fn expensive(value: f64) -> f64 {
    (0..16).fold(value, |acc, _| (acc.abs() + 0.5).sqrt())
}

/// Binary cheap combine.
fn combine_cheap(left: f64, right: f64) -> f64 {
    left + right
}

/// Binary medium combine.
fn combine_medium(left: f64, right: f64) -> f64 {
    left.sin() + right.cos()
}

/// Binary expensive combine.
fn combine_expensive(left: f64, right: f64) -> f64 {
    (0..16).fold(left + right, |acc, _| (acc.abs() + 0.5).sqrt())
}

/// Run the full calibration matrix.
fn calibration(c: &mut Criterion) {
    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    eprintln!(
        "calibration: available_parallelism={parallelism}, sizes={SIZES:?}"
    );

    for &size in &SIZES {
        let input: Array1<f64> = Array1::from_shape_fn(size, |i| (i as f64) * 1e-3);
        let right: Array1<f64> = Array1::from_shape_fn(size, |i| ((i + 1) as f64) * 1e-3);

        for (cost, transform) in [
            ("cheap", cheap as fn(f64) -> f64),
            ("medium", medium),
            ("expensive", expensive),
        ] {
            c.bench_function(
                &format!("elementwise/{cost}/n={size}/seq"),
                |b| b.iter(|| sk_elementwise_transform(&input.view(), transform, 1)),
            );
            c.bench_function(
                &format!("elementwise/{cost}/n={size}/par"),
                |b| b.iter(|| sk_elementwise_transform(&input.view(), transform, parallelism)),
            );
        }

        for (cost, combine) in [
            ("cheap", combine_cheap as fn(f64, f64) -> f64),
            ("medium", combine_medium),
            ("expensive", combine_expensive),
        ] {
            c.bench_function(
                &format!("binary/{cost}/n={size}/seq"),
                |b| b.iter(|| sk_binary_combine(&input.view(), &right.view(), combine, 1)),
            );
            c.bench_function(
                &format!("binary/{cost}/n={size}/par"),
                |b| {
                    b.iter(|| {
                        sk_binary_combine(&input.view(), &right.view(), combine, parallelism)
                    })
                },
            );
        }

        let matrix: Array2<f64> =
            Array2::from_shape_fn((size.div_ceil(64), 64), |(i, j)| (i * 64 + j) as f64 * 1e-3);
        c.bench_function(&format!("axis_sum/axis0/n={size}/seq"), |b| {
            b.iter(|| sk_axis_sum(&matrix.view(), 0, 1))
        });
        c.bench_function(&format!("axis_sum/axis0/n={size}/par"), |b| {
            b.iter(|| sk_axis_sum(&matrix.view(), 0, parallelism))
        });
        c.bench_function(&format!("axis_sum/axis1/n={size}/seq"), |b| {
            b.iter(|| sk_axis_sum(&matrix.view(), 1, 1))
        });
        c.bench_function(&format!("axis_sum/axis1/n={size}/par"), |b| {
            b.iter(|| sk_axis_sum(&matrix.view(), 1, parallelism))
        });

        let mut mutable_matrix = matrix.clone();
        let factor = 1.0 + 1e-9;
        c.bench_function(&format!("scale/n={size}/seq"), |b| {
            b.iter(|| sk_scale_in_place(&mut mutable_matrix, factor, 1))
        });
        let mut mutable_matrix = matrix.clone();
        c.bench_function(&format!("scale/n={size}/par"), |b| {
            b.iter(|| sk_scale_in_place(&mut mutable_matrix, factor, parallelism))
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(300));
    targets = calibration
}
criterion_main!(benches);