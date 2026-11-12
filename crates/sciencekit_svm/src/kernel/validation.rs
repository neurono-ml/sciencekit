//! Input validation shared by `SKSVC` and `SKSVR`
//! (spec `support-vector-machines`).

use std::mem::size_of;
use std::sync::OnceLock;

use sciencekit_common::execution::SKAccessPattern;
use sciencekit_common::{
    SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    SKFloat,
};
use sciencekit_math::{
    SK_KERNEL_PARALLEL_GRAIN, SKKernel, sk_precomputed_kernel_matrix,
    sk_validate_precomputed_query_kernel,
};
use sprs::CompressedStorage;

/// Dimensions `(rows, columns)` of a dense or CSR feature view.
pub fn sk_svm_view_dimensions<F>(view: &SKDataView<'_, F>) -> Result<(usize, usize), SKError> {
    match view {
        SKDataView::Dense(dense) => Ok(dense.dim()),
        SKDataView::Sparse(sparse) => {
            if sparse.storage() != CompressedStorage::CSR {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR",
                });
            }
            Ok((sparse.rows(), sparse.cols()))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: view.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Reject NaN or infinity in a dense or CSR feature view.
pub fn sk_validate_svm_finiteness<F: SKFloat>(view: &SKDataView<'_, F>) -> Result<(), SKError> {
    match view {
        SKDataView::Dense(dense) => {
            if let Some(position) = dense.iter().position(|value| !value.is_finite()) {
                return Err(SKError::NonFiniteValue { position });
            }
            Ok(())
        }
        SKDataView::Sparse(sparse) => {
            let columns = sparse.cols();
            for (row_index, row) in sparse.outer_iterator().enumerate() {
                if let Some((&column_index, _)) = row
                    .indices()
                    .iter()
                    .zip(row.data().iter())
                    .find(|(_, value)| !value.is_finite())
                {
                    return Err(SKError::NonFiniteValue {
                        position: row_index * columns + column_index,
                    });
                }
            }
            Ok(())
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: view.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Validate training features (non-empty, finite, square if precomputed).
pub fn sk_validate_svm_training_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    kernel: SKKernel,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    if kernel == SKKernel::Precomputed {
        let dense = features.as_dense()?;
        let (rows, columns) = dense.dim();
        if rows == 0 || columns == 0 {
            return Err(SKError::EmptyInput { operation });
        }
        sk_precomputed_kernel_matrix(features, rows)?;
        return Ok((rows, columns));
    }
    let (rows, columns) = sk_svm_view_dimensions(&features)?;
    if rows == 0 || columns == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    sk_validate_svm_finiteness(&features)?;
    Ok((rows, columns))
}

/// Validate query features against the fitted feature count and kernel kind.
pub fn sk_validate_svm_query_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    kernel: SKKernel,
    expected_columns: usize,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    if kernel == SKKernel::Precomputed {
        let dense = features.as_dense()?;
        let (rows, columns) = dense.dim();
        if rows == 0 {
            return Err(SKError::EmptyInput { operation });
        }
        sk_validate_precomputed_query_kernel(features, expected_columns)?;
        return Ok((rows, columns));
    }
    let (rows, columns) = sk_svm_view_dimensions(&features)?;
    if rows == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    if columns != expected_columns {
        return Err(SKError::shape_mismatch_2d(
            rows,
            expected_columns,
            rows,
            columns,
        ));
    }
    sk_validate_svm_finiteness(&features)?;
    Ok((rows, columns))
}

fn sk_cached_host_probe() -> (u64, usize) {
    static HOST_PROBE: OnceLock<(u64, usize)> = OnceLock::new();
    *HOST_PROBE.get_or_init(|| {
        let probe = SKExecutionContext::real();
        (probe.available_memory_bytes, probe.cpu_cores)
    })
}

/// Resolve an in-memory random-access execution plan for SVM fit/predict.
pub fn sk_resolve_svm_execution_plan<F: SKFloat>(
    builder_state: &SKBuilderState,
    features: &SKDataView<'_, F>,
    rows_count: usize,
    columns_count: usize,
) -> Result<SKExecutionPlan, SKError> {
    let elements = match features {
        SKDataView::Dense(_) => (rows_count as u64).saturating_mul(columns_count as u64),
        SKDataView::Sparse(sparse) => sparse.nnz() as u64,
        _ => 0,
    };
    let (available_memory_bytes, host_cores) = sk_cached_host_probe();
    let context = SKExecutionContext {
        available_memory_bytes,
        cpu_cores: host_cores.min(rayon::current_num_threads().max(1)),
        dataset_size_bytes: elements.saturating_mul(size_of::<F>() as u64),
        dataset_elements: Some(elements),
        scalar_size_bytes: size_of::<F>() as u64,
        grain: SK_KERNEL_PARALLEL_GRAIN,
        access_pattern: SKAccessPattern::RandomAccess,
        batch_size_hint: None,
        batch_byte_size_hint: None,
        device: None,
    };
    let _ = SKExecutionMode::Automatic;
    builder_state.resolve_plan(&context)
}
