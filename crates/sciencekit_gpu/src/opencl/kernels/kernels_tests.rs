//! Numerical parity tests for the OpenCL kernels against CPU references (spec
//! `opencl-backend`, requirement *OpenCL kernels*).
//!
//! Every test runs on a real device when the host has one: it uploads a
//! deterministic matrix, runs the kernel, downloads the result and compares it
//! with a CPU reference computed here. Small, large, unaligned and degenerate
//! shapes are covered, plus a deliberate compile failure that must carry the
//! driver's build log.

use ndarray::{Array2, array};
use sciencekit_common::SKError;

use super::{
    SKOpenClDistanceMetric, run_combine, run_gemm, run_pairwise_distance, run_transform,
    run_tree_predict,
};
use crate::opencl::SKOpenClBackend;
use crate::opencl::buffer::{allocate_resident, download_buffer, upload_view};
use crate::opencl::session::SKOpenClSession;

/// The selected device's session, or a printed skip on a host without a device.
fn real_session() -> Option<SKOpenClBackend> {
    let backend = SKOpenClBackend::builder().build_or_fallback();
    if !backend.is_available() {
        println!("[opencl] no device available; kernel tests skip");
        return None;
    }
    let device = backend.device().unwrap();
    println!(
        "[opencl] kernel parity runs on device {}: `{}`",
        device.index, device.name
    );
    Some(backend)
}

fn session(backend: &SKOpenClBackend) -> &SKOpenClSession {
    backend.session().expect("a device session")
}

fn deterministic_matrix(rows: usize, columns: usize, seed: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, columns), |(row, column)| {
        (((row * 13 + column * 7) % 23) as f64) * 0.25 - 2.0 + seed
    })
}

fn reference_pairwise(
    left: &Array2<f64>,
    right: &Array2<f64>,
    metric: SKOpenClDistanceMetric,
) -> Array2<f64> {
    Array2::from_shape_fn((left.nrows(), right.nrows()), |(row, other)| {
        let accumulator = left.row(row).iter().zip(right.row(other).iter()).fold(
            0.0_f64,
            |accumulator, (left_value, right_value)| {
                let difference = left_value - right_value;
                match metric {
                    SKOpenClDistanceMetric::SquaredEuclidean => {
                        accumulator + difference * difference
                    }
                    SKOpenClDistanceMetric::Euclidean => accumulator + difference * difference,
                    SKOpenClDistanceMetric::Manhattan => accumulator + difference.abs(),
                    SKOpenClDistanceMetric::Chebyshev => accumulator.max(difference.abs()),
                }
            },
        );
        match metric {
            SKOpenClDistanceMetric::Euclidean => accumulator.sqrt(),
            _ => accumulator,
        }
    })
}

fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    let worst = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        worst <= tolerance,
        "largest deviation {worst} exceeds the tolerance {tolerance}"
    );
}

/// All four metrics match the CPU reference on a small unaligned input.
#[test]
fn pairwise_metrics_match_the_cpu_reference() {
    let Some(backend) = real_session() else {
        return;
    };
    let session = session(&backend);
    let left = deterministic_matrix(13, 11, 0.0);
    let right = deterministic_matrix(17, 11, 0.5);
    let left_resident = upload_view(session, left.view()).unwrap();
    let right_resident = upload_view(session, right.view()).unwrap();
    let output = allocate_resident::<f64>(session, (13, 17)).unwrap();

    for metric in [
        SKOpenClDistanceMetric::SquaredEuclidean,
        SKOpenClDistanceMetric::Euclidean,
        SKOpenClDistanceMetric::Manhattan,
        SKOpenClDistanceMetric::Chebyshev,
    ] {
        run_pairwise_distance(session, &left_resident, &right_resident, &output, metric).unwrap();
        let actual = download_buffer(session, &output).unwrap();
        assert_close(&actual, &reference_pairwise(&left, &right, metric), 1e-9);
        println!("[opencl] `{}` matched the CPU reference", metric.name());
    }
}

/// Large pairwise inputs still match the CPU reference.
#[test]
fn large_pairwise_matches_the_cpu_reference() {
    let Some(backend) = real_session() else {
        return;
    };
    let session = session(&backend);
    let left = deterministic_matrix(257, 33, 0.0);
    let right = deterministic_matrix(129, 33, 1.0);
    let left_resident = upload_view(session, left.view()).unwrap();
    let right_resident = upload_view(session, right.view()).unwrap();
    let output = allocate_resident::<f64>(session, (257, 129)).unwrap();
    run_pairwise_distance(
        session,
        &left_resident,
        &right_resident,
        &output,
        SKOpenClDistanceMetric::Euclidean,
    )
    .unwrap();
    let actual = download_buffer(session, &output).unwrap();
    assert_close(
        &actual,
        &reference_pairwise(&left, &right, SKOpenClDistanceMetric::Euclidean),
        1e-8,
    );
}

/// Single-precision pairwise distances keep the documented tolerance.
#[test]
fn single_precision_pairwise_matches_within_tolerance() {
    let Some(backend) = real_session() else {
        return;
    };
    let session = session(&backend);
    let left: Array2<f32> =
        Array2::from_shape_fn((9, 4), |(row, column)| ((row * 5 + column) as f32) * 0.5);
    let right: Array2<f32> =
        Array2::from_shape_fn((6, 4), |(row, column)| ((row * 3 + column) as f32) * 0.25);
    let left_resident = upload_view(session, left.view()).unwrap();
    let right_resident = upload_view(session, right.view()).unwrap();
    let output = allocate_resident::<f32>(session, (9, 6)).unwrap();
    run_pairwise_distance(
        session,
        &left_resident,
        &right_resident,
        &output,
        SKOpenClDistanceMetric::SquaredEuclidean,
    )
    .unwrap();
    let actual = download_buffer(session, &output).unwrap();
    let expected: Array2<f32> = Array2::from_shape_fn((9, 6), |(row, other)| {
        left.row(row)
            .iter()
            .zip(right.row(other).iter())
            .map(|(left_value, right_value)| {
                (left_value - right_value) * (left_value - right_value)
            })
            .sum()
    });
    let worst = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst <= 1e-3, "single-precision deviation {worst}");
}

/// The tiled GEMM matches a CPU matrix multiplication, small and large.
#[test]
fn gemm_matches_the_cpu_reference() {
    let Some(backend) = real_session() else {
        return;
    };
    let session = session(&backend);

    for (rows, shared, columns) in [(17, 23, 19), (64, 96, 32), (129, 47, 65)] {
        let left = deterministic_matrix(rows, shared, 0.0);
        let right = deterministic_matrix(shared, columns, 0.25);
        let left_resident = upload_view(session, left.view()).unwrap();
        let right_resident = upload_view(session, right.view()).unwrap();
        let output = allocate_resident::<f64>(session, (rows, columns)).unwrap();
        run_gemm(session, &left_resident, &right_resident, &output).unwrap();
        let actual = download_buffer(session, &output).unwrap();

        let mut expected = Array2::<f64>::zeros((rows, columns));
        ndarray::linalg::general_mat_mul(1.0, &left, &right, 0.0, &mut expected);
        assert_close(&actual, &expected, 1e-9);
        println!("[opencl] gemm {rows}x{shared}x{columns} matched the CPU reference");
    }
}

/// The affine transform and the elementwise combination match the CPU.
#[test]
fn elementwise_kernels_match_the_cpu_reference() {
    let Some(backend) = real_session() else {
        return;
    };
    let session = session(&backend);
    let input = deterministic_matrix(37, 29, 0.0);
    let other = deterministic_matrix(37, 29, 2.0);

    let input_resident = upload_view(session, input.view()).unwrap();
    let other_resident = upload_view(session, other.view()).unwrap();
    let output = allocate_resident::<f64>(session, (37, 29)).unwrap();

    run_transform(session, &input_resident, &output, 2.5, -0.75).unwrap();
    let transformed = download_buffer(session, &output).unwrap();
    let expected = input.mapv(|value| value * 2.5 - 0.75);
    assert_close(&transformed, &expected, 1e-9);

    run_combine(session, &input_resident, &other_resident, &output).unwrap();
    let combined = download_buffer(session, &output).unwrap();
    assert_close(&combined, &(&input + &other), 1e-9);
}

/// Tree prediction walks the same nodes the CPU reference walks.
#[test]
fn tree_prediction_matches_the_cpu_reference() {
    let Some(backend) = real_session() else {
        return;
    };
    let session = session(&backend);
    let features: Array2<f64> =
        array![[0.2, 0.9], [0.7, -0.5], [0.7, 0.4], [0.1, -3.0], [0.9, 0.0],];
    // feature, threshold, left, right, leaf value, leaf flag
    let nodes: Array2<f64> = array![
        [0.0, 0.5, 1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
        [1.0, 0.0, 3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 3.0, 1.0],
    ];
    let feature_resident = upload_view(session, features.view()).unwrap();
    let node_resident = upload_view(session, nodes.view()).unwrap();
    let output = allocate_resident::<f64>(session, (features.nrows(), 1)).unwrap();
    run_tree_predict(session, &feature_resident, &node_resident, &output).unwrap();
    let actual = download_buffer(session, &output).unwrap();
    // The last row is `0.9 > 0.5` then `0.0 <= 0.0`, so it reaches leaf `2`.
    let expected: Array2<f64> = array![[-1.0], [2.0], [3.0], [-1.0], [2.0]];
    assert_eq!(actual, expected);
}

/// A program that cannot compile reports the driver's build log.
#[test]
fn a_broken_program_reports_the_build_log() {
    let Some(backend) = real_session() else {
        return;
    };
    let session = session(&backend);
    let error = session
        .program("broken-program", "__kernel void sk_broken(")
        .expect_err("the source cannot compile");
    match error {
        SKError::Conversion(message) => {
            assert!(
                message.contains("build failed"),
                "unexpected message: {message}"
            );
            assert!(
                message.to_lowercase().contains("error"),
                "the driver build log must be surfaced: {message}"
            );
            println!(
                "[opencl] build log surfaced: {}",
                message.lines().next().unwrap_or("")
            );
        }
        other => panic!("unexpected error: {other}"),
    }
}

/// One backend must serve both precisions: the program cache is per scalar type,
/// so an `f32` dispatch cannot leave an `f32` program behind for a later `f64`
/// dispatch of the same kernel.
#[test]
fn one_backend_serves_both_precisions_of_the_same_kernel() {
    let Some(backend) = real_session() else {
        return;
    };
    let session = session(&backend);

    let left_single: Array2<f32> = Array2::from_shape_fn((7, 5), |(row, column)| {
        ((row * 3 + column) as f32) * 0.5 - 1.0
    });
    let right_single: Array2<f32> =
        Array2::from_shape_fn((11, 5), |(row, column)| ((row * 7 + column) as f32) * 0.25);
    let left_resident = upload_view(session, left_single.view()).unwrap();
    let right_resident = upload_view(session, right_single.view()).unwrap();
    let single_output = allocate_resident::<f32>(session, (7, 11)).unwrap();
    run_pairwise_distance(
        session,
        &left_resident,
        &right_resident,
        &single_output,
        SKOpenClDistanceMetric::SquaredEuclidean,
    )
    .unwrap();
    let single_actual = download_buffer(session, &single_output).unwrap();
    let single_expected: Array2<f32> = Array2::from_shape_fn((7, 11), |(row, other)| {
        left_single
            .row(row)
            .iter()
            .zip(right_single.row(other).iter())
            .map(|(left_value, right_value)| (left_value - right_value).powi(2))
            .sum()
    });
    let single_worst = single_actual
        .iter()
        .zip(single_expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        single_worst <= 1e-3,
        "the f32 dispatch must match the CPU: {single_worst}"
    );

    let left_double = deterministic_matrix(7, 5, 0.0);
    let right_double = deterministic_matrix(11, 5, 0.5);
    let left_resident = upload_view(session, left_double.view()).unwrap();
    let right_resident = upload_view(session, right_double.view()).unwrap();
    let double_output = allocate_resident::<f64>(session, (7, 11)).unwrap();
    run_pairwise_distance(
        session,
        &left_resident,
        &right_resident,
        &double_output,
        SKOpenClDistanceMetric::SquaredEuclidean,
    )
    .unwrap();
    let double_actual = download_buffer(session, &double_output).unwrap();
    assert_close(
        &double_actual,
        &reference_pairwise(
            &left_double,
            &right_double,
            SKOpenClDistanceMetric::SquaredEuclidean,
        ),
        1e-9,
    );
}
