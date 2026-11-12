//! Per-pair binary C-SVC fitting and feature-subset extraction
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_math::{SKKernel, SKKernelParameters};

use super::core_implementation::SKSVCConfiguration;
use super::model::SKBinarySVCModel;
use crate::kernel::SKOwnedFeatureBlock;
use crate::probability::sk_calibrate_binary_platt;
use crate::smo::{SKSmoOptions, sk_solve_binary_c_svc};

/// Extract a square subset Gram matrix when `kernel == Precomputed`, or select
/// the subset of rows in their native dense/CSR representation.
pub(crate) fn sk_extract_subset_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    kernel: SKKernel,
    subset_indices: &[usize],
) -> Result<SKOwnedFeatureBlock<F>, SKError> {
    if kernel == SKKernel::Precomputed {
        let dense = features.as_dense()?;
        let count = subset_indices.len();
        let subset_gram = Array2::from_shape_fn((count, count), |(row, column)| {
            dense[[subset_indices[row], subset_indices[column]]]
        });
        return Ok(SKOwnedFeatureBlock::Dense(subset_gram));
    }
    SKOwnedFeatureBlock::select_rows(features, subset_indices)
}

/// Fit one binary `(class_negative, class_positive)` C-SVC problem.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_fit_binary_pair<F: SKFloat>(
    configuration: &SKSVCConfiguration<F>,
    features: SKDataView<'_, F>,
    class_indices: &[usize],
    per_class_weights: &[F],
    kernel_parameters: SKKernelParameters<F>,
    class_negative: usize,
    class_positive: usize,
    pair_index: u64,
) -> Result<SKBinarySVCModel<F>, SKError> {
    let mut pair_rows = Vec::new();
    let mut signs = Vec::new();
    let mut upper_bounds = Vec::new();
    for (row_index, &label) in class_indices.iter().enumerate() {
        if label == class_negative {
            pair_rows.push(row_index);
            signs.push(-1_i8);
            upper_bounds.push(configuration.regularization_c * per_class_weights[class_negative]);
        } else if label == class_positive {
            pair_rows.push(row_index);
            signs.push(1_i8);
            upper_bounds.push(configuration.regularization_c * per_class_weights[class_positive]);
        }
    }

    let subset = sk_extract_subset_features(features, kernel_parameters.kernel, &pair_rows)?;
    let options = SKSmoOptions {
        tolerance: configuration.tolerance,
        maximum_number_of_iterations: configuration.maximum_number_of_iterations,
        cache_size: configuration.cache_size,
        shrinking: configuration.shrinking,
    };
    let solved = sk_solve_binary_c_svc(
        &signs,
        &upper_bounds,
        subset.view(),
        kernel_parameters,
        options,
    )?;

    let platt_parameters = if configuration.probability {
        Some(sk_calibrate_binary_platt(
            subset.view(),
            &signs,
            &upper_bounds,
            kernel_parameters,
            options,
            configuration.random_seed,
            pair_index,
        )?)
    } else {
        None
    };

    let support_training_indices: Vec<usize> = solved
        .support_indices
        .iter()
        .map(|&local| pair_rows[local])
        .collect();
    let support_vectors = if kernel_parameters.kernel == SKKernel::Precomputed {
        SKOwnedFeatureBlock::Dense(Array2::zeros((support_training_indices.len(), 0)))
    } else {
        SKOwnedFeatureBlock::select_rows(features, &support_training_indices)?
    };

    Ok(SKBinarySVCModel {
        class_negative,
        class_positive,
        support_vectors,
        support_training_indices,
        dual_coefficients: Array1::from(solved.dual_coefficients),
        bias: solved.bias,
        iterations: solved.iterations,
        platt_parameters,
    })
}
