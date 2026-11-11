//! Companion tests for the determinism and the thread safety of the randomized
//! range finder (spec `randomized-svd`).

use ndarray::Array2;
use sciencekit_common::SKDataView;

use super::dense_tests::fixture_low_rank_matrix;
use super::randomized::{SKRandomizedSvdBuilder, sk_randomized_svd};
use super::result::SKRandomizedSvdResult;

/// The result type is a plain owned value: it can be moved between threads.
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn randomized_concurrency_result_is_send_and_sync() {
    assert_send_sync::<SKRandomizedSvdResult<f64>>();
    assert_send_sync::<SKRandomizedSvdBuilder>();
}

#[test]
fn randomized_concurrency_repeated_calls_are_identical() {
    let matrix = fixture_low_rank_matrix(40, 12, 3, 51);
    let first = sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 17, None)
        .expect("the decomposition must succeed");
    let second = sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 17, None)
        .expect("the decomposition must succeed");
    assert_eq!(first.u, second.u);
    assert_eq!(first.singular_values, second.singular_values);
    assert_eq!(first.v, second.v);
}

#[test]
fn randomized_concurrency_concurrent_calls_agree_with_the_single_threaded_result() {
    let matrix = fixture_low_rank_matrix(40, 12, 3, 53);
    let expected = sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 19, None)
        .expect("the decomposition must succeed");

    let results: Vec<SKRandomizedSvdResult<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let matrix = &matrix;
                scope.spawn(move || {
                    sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 19, None)
                        .expect("the decomposition must succeed")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("the worker must not panic"))
            .collect()
    });

    for result in results {
        assert_eq!(result.u, expected.u);
        assert_eq!(result.singular_values, expected.singular_values);
        assert_eq!(result.v, expected.v);
    }
}

#[test]
fn randomized_concurrency_does_not_depend_on_the_thread_count() {
    let matrix: Array2<f64> = fixture_low_rank_matrix(30, 10, 3, 59);
    let expected = sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 23, None)
        .expect("the decomposition must succeed");

    // The same factorization, computed with a different number of worker
    // threads available to the process.
    let workers = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let matrix = &matrix;
            let handle = scope.spawn(move || {
                sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 23, None)
                    .expect("the decomposition must succeed")
            });
            let result = handle.join().expect("the worker must not panic");
            assert_eq!(result.u, expected.u);
            assert_eq!(result.singular_values, expected.singular_values);
            assert_eq!(result.v, expected.v);
        }
    });
}
