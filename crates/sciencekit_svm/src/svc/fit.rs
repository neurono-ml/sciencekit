//! `SKSupervisedFit` implementation for `SKSVC`
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};
use sciencekit_math::{SKKernel, SKKernelParameters, sk_resolve_kernel_parameters};

use super::core_implementation::{SKSVC, SKSVCConfiguration};
use super::model::{SKBinarySVCModel, SKSVCModel};
use super::targets::sk_canonicalize_svc_targets;
use super::weights::sk_resolve_class_weights;
use crate::kernel::{
    SKOwnedFeatureBlock, sk_svm_view_dimensions, sk_validate_svm_training_features,
};
use crate::smo::{SKSmoOptions, sk_solve_binary_c_svc};

impl<F: SKFloat> SKSupervisedFit<F> for SKSVC<F> {
    type Model = SKSVCModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let targets = targets.try_into()?;
        let (rows_count, columns_count) =
            sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || sk_fit_svc(self, features, &targets))
    }
}

/// Extract a square sub-Gram matrix when `kernel == Precomputed`, or select
/// the subset of rows in their native dense/CSR representation.
pub(crate) fn sk_extract_subset_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    kernel: SKKernel,
    subset_indices: &[usize],
) -> Result<SKOwnedFeatureBlock<F>, SKError> {
    if kernel == SKKernel::Precomputed {
        let dense = features.as_dense()?;
        let count = subset_indices.len();
        let sub_gram =
            Array2::from_shape_fn((count, count), |(r, c)| dense[[subset_indices[r], subset_indices[c]]]);
        return Ok(SKOwnedFeatureBlock::Dense(sub_gram));
    }
    SKOwnedFeatureBlock::select_rows(features, subset_indices)
}

/// Fit one binary `(class_negative, class_positive)` C-SVC problem.
pub(crate) fn sk_fit_binary_pair<F: SKFloat>(
    configuration: &SKSVCConfiguration<F>,
    features: SKDataView<'_, F>,
    class_indices: &[usize],
    per_class_weights: &[F],
    kernel_parameters: SKKernelParameters<F>,
    class_negative: usize,
    class_positive: usize,
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
        cache_size_mib: configuration.cache_size,
        shrinking: configuration.shrinking,
    };
    let solved = sk_solve_binary_c_svc(
        &signs,
        &upper_bounds,
        subset.view(),
        kernel_parameters,
        options,
    )?;

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
    })
}

fn sk_fit_svc<F: SKFloat>(
    estimator: &SKSVC<F>,
    features: SKDataView<'_, F>,
    targets: &SKTargetView<'_, F>,
) -> Result<SKSVCModel<F>, SKError> {
    let config = &estimator.configuration;
    let (rows, columns) = sk_validate_svm_training_features(features, config.kernel, "fit")?;
    let (class_indices, classes) = sk_canonicalize_svc_targets(targets, rows)?;
    let number_of_classes = classes.number_of_classes();
    if number_of_classes > 2 {
        return Err(SKError::UnsupportedOperation {
            operation: "svc_multiclass",
            reason: "one-vs-one multiclass arrives in task 3.3".to_owned(),
        });
    }
    let per_class_weights =
        sk_resolve_class_weights(&config.class_weight, &class_indices, number_of_classes)?;
    let kernel_parameters = sk_resolve_kernel_parameters(
        config.kernel,
        config.gamma,
        config.degree,
        config.coef0,
        features,
    )?;

    let binary = sk_fit_binary_pair(
        config,
        features,
        &class_indices,
        &per_class_weights,
        kernel_parameters,
        0,
        1,
    )?;
    Ok(SKSVCModel {
        classes,
        pairwise_models: vec![binary],
        kernel_parameters,
        number_of_features: columns,
        backend: config.backend,
        execution_mode: config.execution_mode,
        probability_enabled: config.probability,
    })
}
