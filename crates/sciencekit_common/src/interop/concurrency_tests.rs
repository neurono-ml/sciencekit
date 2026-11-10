//! Tests for read-only concurrency of exported states
//! (spec `model-export-contracts`).

use std::sync::Arc;
use std::thread;

use super::mock_model::{SKMockLinearModel, mock_linear_model};
use super::{SKExportable, SKModelMetadata, SKModelState, SKParameterValue, SKTensor};

/// A compile-time assertion that a type can be shared across threads.
fn assert_send_sync<T: Send + Sync>() {}

/// Every exported contract type is immutable and safely shareable.
#[test]
fn interop_concurrency_contracts_are_send_and_sync() {
    assert_send_sync::<SKModelState<f32>>();
    assert_send_sync::<SKModelState<f64>>();
    assert_send_sync::<SKTensor<f64>>();
    assert_send_sync::<SKModelMetadata>();
    assert_send_sync::<SKParameterValue>();
    assert_send_sync::<SKMockLinearModel<f64>>();
}

/// Concurrent export of one model returns exactly the single-threaded state.
#[test]
fn interop_concurrency_concurrent_exports_agree() {
    let model = Arc::new(mock_linear_model());
    let expected = model.export_state().unwrap();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);
            thread::spawn(move || model.export_state().unwrap())
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
}

/// A shared exported state serves concurrent reads of the same tensor.
#[test]
fn interop_concurrency_shared_state_reads_agree() {
    let state = Arc::new(mock_linear_model().export_state().unwrap());

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let state = Arc::clone(&state);
            thread::spawn(move || {
                state
                    .tensor("linear.coefficients")
                    .map(|tensor| tensor.number_of_elements())
                    .unwrap()
            })
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().unwrap(), 3);
    }
}
