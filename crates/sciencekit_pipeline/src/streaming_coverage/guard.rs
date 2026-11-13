//! The dense-only guard fixture and its tests.

use sciencekit_common::{SKDataView, SKError};

/// A deliberately dense-only streaming estimator must reject CSR up front with
/// `UnsupportedRepresentation`, a named representation and a suggestion, before
/// processing any element.
#[test]
fn sparse_streaming_guard_rejects_csr_before_processing() {
    let sparse = sprs::CsMat::new((2, 2), vec![0, 1, 1], vec![0], vec![1.0_f64]);
    let error = sk_dense_only_streaming_fixture(SKDataView::Sparse(sparse.view()))
        .expect_err("a dense-only streaming path must reject CSR");
    match error {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "csr");
            assert!(
                suggestion.contains("dense"),
                "the suggestion must name the conversion path"
            );
        }
        other => panic!("expected an unsupported-representation error, got {other:?}"),
    }
}

/// The guard accepts the dense representation.
#[test]
fn sparse_streaming_guard_accepts_dense_blocks() {
    let dense = ndarray::array![[1.0_f64, 0.0], [0.0, 2.0]];
    let rows = sk_dense_only_streaming_fixture(SKDataView::Dense(dense.view()))
        .expect("the dense path is supported");
    assert_eq!(rows, 2);
}

/// A stand-in for a dense-only streaming estimator: the representation is
/// matched once, before any element is read.
fn sk_dense_only_streaming_fixture(features: SKDataView<'_, f64>) -> Result<usize, SKError> {
    match features {
        SKDataView::Dense(dense) => Ok(dense.nrows()),
        SKDataView::Sparse(_) => Err(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "convert the block to dense with to_dense, or use a sparse-aware estimator",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}
