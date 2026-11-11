//! `SKSupervisedFit` implementation for `SKSVR`
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};
use sciencekit_math::{SKKernel, sk_resolve_kernel_parameters};

use super::core_implementation::SKSVR;
use super::model::SKSVRModel;
use crate::kernel::{
    SKOwnedFeatureBlock, sk_svm_view_dimensions, sk_validate_svm_training_features,
};
use crate::smo::{SKSmoOptions, sk_solve_epsilon_svr};

impl<F: SKFloat> SKSupervisedFit<F> for SKSVR<F> {
    type Model = SKSVRModel<F>;
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
        sk_run_operation(attributes, || sk_fit_svr(self, features, &targets))
    }
}

fn sk_fit_svr<F: SKFloat>(
    estimator: &SKSVR<F>,
    features: SKDataView<'_, F>,
    targets: &SKTargetView<'_, F>,
) -> Result<SKSVRModel<F>, SKError> {
    let config = &estimator.configuration;
    let (rows, columns) = sk_validate_svm_training_features(features, config.kernel, "fit")?;
    let responses = targets.as_continuous()?;
    if responses.len() != rows {
        return Err(SKError::shape_mismatch_2d(rows, 1, responses.len(), 1));
    }
    if let Some(position) = responses.iter().position(|value| !value.is_finite()) {
        return Err(SKError::NonFiniteValue { position });
    }

    let kernel_parameters = sk_resolve_kernel_parameters(
        config.kernel,
        config.gamma,
        config.degree,
        config.coef0,
        features,
    )?;
    let options = SKSmoOptions {
        tolerance: config.tolerance,
        maximum_number_of_iterations: config.maximum_number_of_iterations,
        cache_size_mib: config.cache_size,
        shrinking: config.shrinking,
    };
    let target_vec: Vec<F> = responses.iter().copied().collect();
    let solved = sk_solve_epsilon_svr(
        &target_vec,
        config.regularization_c,
        config.epsilon,
        features,
        kernel_parameters,
        options,
    )?;

    let support_vectors = if kernel_parameters.kernel == SKKernel::Precomputed {
        SKOwnedFeatureBlock::Dense(Array2::zeros((solved.support_indices.len(), 0)))
    } else {
        SKOwnedFeatureBlock::select_rows(features, &solved.support_indices)?
    };
    Ok(SKSVRModel {
        support_vectors,
        support_training_indices: solved.support_indices,
        dual_coefficients: Array1::from(solved.dual_coefficients),
        bias: solved.bias,
        iterations: solved.iterations,
        kernel_parameters,
        number_of_features: columns,
        backend: config.backend,
        execution_mode: config.execution_mode,
    })
}
