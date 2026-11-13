//! Shared state encoding for the support-vector models.
//!
//! Support vectors keep their training representation: a dense block travels as
//! a two-dimensional floating tensor, a CSR block as its `indptr`/`indices`/
//! `data` triple and is rebuilt as CSR — never densified.

use ndarray::Array2;
use sciencekit_common::{
    SKError, SKFloat, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    sk_read_floating_tensor, sk_read_integer_tensor, sk_string_parameter,
};
use sciencekit_math::{SKKernel, SKKernelParameters};
use sprs::CsMat;

use crate::kernel::SKOwnedFeatureBlock;

/// The metadata parameter naming the support-vector representation.
pub const SK_SUPPORT_VECTOR_REPRESENTATION: &str = "support_vector_representation";

/// Encode a support-vector block under a name prefix.
pub(crate) fn export_support_vectors<F: SKFloat>(
    block: &SKOwnedFeatureBlock<F>,
    prefix: &str,
) -> Result<(Vec<SKTensor<F>>, &'static str), SKError> {
    match block {
        SKOwnedFeatureBlock::Dense(dense) => Ok((
            vec![SKTensor::floating(
                format!("{prefix}.support_vectors_dense"),
                vec![dense.nrows(), dense.ncols()],
                dense.iter().copied().collect(),
            )?],
            "dense",
        )),
        SKOwnedFeatureBlock::Sparse(sparse) => {
            let indptr: Vec<i64> = sparse
                .indptr()
                .raw_storage()
                .iter()
                .map(|&value| value as i64)
                .collect();
            let indices: Vec<i64> = sparse.indices().iter().map(|&value| value as i64).collect();
            let data = sparse.data().to_vec();
            Ok((
                vec![
                    SKTensor::integer(
                        format!("{prefix}.support_vectors_indptr"),
                        vec![sparse.rows() + 1],
                        indptr,
                    )?,
                    SKTensor::integer(
                        format!("{prefix}.support_vectors_indices"),
                        vec![indices.len()],
                        indices,
                    )?,
                    SKTensor::floating(
                        format!("{prefix}.support_vectors_data"),
                        vec![data.len()],
                        data,
                    )?,
                ],
                "csr",
            ))
        }
    }
}

/// Rebuild a support-vector block from a state.
pub(crate) fn import_support_vectors<F: SKFloat>(
    state: &SKModelState<F>,
    prefix: &str,
    representation: &str,
) -> Result<SKOwnedFeatureBlock<F>, SKError> {
    if representation == "csr" {
        let indptr: Vec<usize> =
            sk_read_integer_tensor(state, &format!("{prefix}.support_vectors_indptr"))?
                .into_iter()
                .map(|value| value.max(0) as usize)
                .collect();
        let indices: Vec<usize> =
            sk_read_integer_tensor(state, &format!("{prefix}.support_vectors_indices"))?
                .into_iter()
                .map(|value| value.max(0) as usize)
                .collect();
        let (_, data) = sk_read_floating_tensor(state, &format!("{prefix}.support_vectors_data"))?;
        let rows = indptr.len().saturating_sub(1);
        let columns = state.metadata().feature_count();
        Ok(SKOwnedFeatureBlock::Sparse(CsMat::new(
            (rows, columns),
            indptr,
            indices,
            data,
        )))
    } else {
        let (shape, values) =
            sk_read_floating_tensor(state, &format!("{prefix}.support_vectors_dense"))?;
        let &[rows, columns] = shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, state.metadata().feature_count()],
                found: shape,
            });
        };
        let dense = Array2::from_shape_vec((rows, columns), values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        Ok(SKOwnedFeatureBlock::Dense(dense))
    }
}

/// Attach the resolved kernel parameters to metadata.
pub(crate) fn with_kernel_parameters<F: SKFloat>(
    mut metadata: SKModelMetadata,
    parameters: &SKKernelParameters<F>,
) -> Result<SKModelMetadata, SKError> {
    metadata = metadata
        .with_parameter(
            "kernel",
            SKParameterValue::from(kernel_name(parameters.kernel)),
        )?
        .with_parameter(
            "gamma",
            SKParameterValue::from(parameters.gamma.to_f64().unwrap_or(0.0)),
        )?
        .with_parameter("degree", SKParameterValue::from(parameters.degree as i64))?
        .with_parameter(
            "coef0",
            SKParameterValue::from(parameters.coef0.to_f64().unwrap_or(0.0)),
        )?;
    Ok(metadata)
}

/// Read the resolved kernel parameters from metadata.
pub(crate) fn read_kernel_parameters<F: SKFloat>(
    metadata: &SKModelMetadata,
) -> Result<SKKernelParameters<F>, SKError> {
    Ok(SKKernelParameters::new(
        kernel_from_name(sk_string_parameter(metadata, "kernel", "rbf")),
        F::from(
            metadata
                .parameter("gamma")
                .and_then(SKParameterValue::as_float)
                .unwrap_or(0.0),
        )
        .ok_or(SKError::Conversion("gamma is not finite".to_string()))?,
        metadata
            .parameter("degree")
            .and_then(SKParameterValue::as_integer)
            .unwrap_or(3)
            .max(1) as u32,
        F::from(
            metadata
                .parameter("coef0")
                .and_then(SKParameterValue::as_float)
                .unwrap_or(0.0),
        )
        .ok_or(SKError::Conversion("coef0 is not finite".to_string()))?,
    ))
}

fn kernel_name(kernel: SKKernel) -> &'static str {
    match kernel {
        SKKernel::Linear => "linear",
        SKKernel::Polynomial => "polynomial",
        SKKernel::RadialBasisFunction => "rbf",
        SKKernel::Sigmoid => "sigmoid",
        SKKernel::Precomputed => "precomputed",
    }
}

fn kernel_from_name(name: &str) -> SKKernel {
    match name {
        "linear" => SKKernel::Linear,
        "polynomial" => SKKernel::Polynomial,
        "sigmoid" => SKKernel::Sigmoid,
        "precomputed" => SKKernel::Precomputed,
        _ => SKKernel::RadialBasisFunction,
    }
}
