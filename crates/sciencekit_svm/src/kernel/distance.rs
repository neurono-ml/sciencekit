//! Single-row and diagonal kernel evaluation over dense and CSR views
//! (spec `support-vector-machines`).

use ndarray::Axis;
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_math::{SKKernel, SKKernelParameters, sk_kernel_matrix};
use sprs::{CompressedStorage, CsMatView};

use super::validation::sk_svm_view_dimensions;

/// Compute the kernel diagonal `K(x_i, x_i)` for every row of `features`
/// without materializing the `n × n` matrix.
pub fn sk_evaluate_kernel_diagonal<F: SKFloat>(
    features: SKDataView<'_, F>,
    parameters: &SKKernelParameters<F>,
) -> Result<Vec<F>, SKError> {
    if parameters.kernel == SKKernel::Precomputed {
        let dense = features.as_dense()?;
        return Ok((0..dense.nrows())
            .map(|row_index| dense[[row_index, row_index]])
            .collect());
    }
    if parameters.kernel == SKKernel::RadialBasisFunction {
        let (rows, _) = sk_svm_view_dimensions(&features)?;
        return Ok(vec![F::one(); rows]);
    }
    let squared_norms: Vec<F> = match features {
        SKDataView::Dense(dense) => dense
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .fold(F::zero(), |accumulator, &value| accumulator + value * value)
            })
            .collect(),
        SKDataView::Sparse(sparse) => {
            if sparse.storage() != CompressedStorage::CSR {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR",
                });
            }
            sparse
                .outer_iterator()
                .map(|row| {
                    row.data()
                        .iter()
                        .fold(F::zero(), |accumulator, &value| accumulator + value * value)
                })
                .collect()
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a dense matrix or a CSR matrix",
            });
        }
    };
    Ok(squared_norms
        .into_iter()
        .map(|squared_norm| match parameters.kernel {
            SKKernel::Linear => squared_norm,
            SKKernel::Polynomial => {
                (parameters.gamma * squared_norm + parameters.coef0).powi(parameters.degree as i32)
            }
            SKKernel::Sigmoid => (parameters.gamma * squared_norm + parameters.coef0).tanh(),
            SKKernel::RadialBasisFunction | SKKernel::Precomputed => F::one(),
        })
        .collect())
}

/// Evaluate a single kernel row `K(x_{row_index}, X)` on demand without
/// copying or densifying `features`.
pub fn sk_evaluate_kernel_row<F: SKFloat>(
    features: SKDataView<'_, F>,
    row_index: usize,
    parameters: &SKKernelParameters<F>,
) -> Result<Vec<F>, SKError> {
    if parameters.kernel == SKKernel::Precomputed {
        let dense = features.as_dense()?;
        return Ok(dense.row(row_index).to_vec());
    }
    match features {
        SKDataView::Dense(dense) => {
            let single = dense.row(row_index).insert_axis(Axis(0));
            let matrix = sk_kernel_matrix(parameters, SKDataView::Dense(single), features)?;
            Ok(matrix.row(0).to_vec())
        }
        SKDataView::Sparse(sparse) => {
            if sparse.storage() != CompressedStorage::CSR {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR",
                });
            }
            let range = sparse.indptr().outer_inds_sz(row_index);
            let indices = &sparse.indices()[range.clone()];
            let data = &sparse.data()[range];
            let indptr = [0usize, data.len()];
            let single = CsMatView::new((1, sparse.cols()), &indptr, indices, data);
            let matrix = sk_kernel_matrix(parameters, SKDataView::Sparse(single), features)?;
            Ok(matrix.row(0).to_vec())
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}
