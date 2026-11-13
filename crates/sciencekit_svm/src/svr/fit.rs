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
    SKOwnedFeatureBlock, sk_resolve_svm_execution_plan, sk_svm_view_dimensions,
    sk_validate_svm_training_features,
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
        let (rows_count, columns_count) = sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
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
    let configuration = &estimator.configuration;
    let (rows, columns) = sk_validate_svm_training_features(features, configuration.kernel, "fit")?;
    let _plan =
        sk_resolve_svm_execution_plan(&configuration.builder_state, &features, rows, columns)?;
    let responses = targets.as_continuous()?;
    if responses.len() != rows {
        return Err(SKError::shape_mismatch_2d(rows, 1, responses.len(), 1));
    }
    if let Some(position) = responses.iter().position(|value| !value.is_finite()) {
        return Err(SKError::NonFiniteValue { position });
    }

    let kernel_parameters = sk_resolve_kernel_parameters(
        configuration.kernel,
        configuration.gamma,
        configuration.degree,
        configuration.coef0,
        features,
    )?;
    let options = SKSmoOptions {
        tolerance: configuration.tolerance,
        maximum_number_of_iterations: configuration.maximum_number_of_iterations,
        cache_size: configuration.cache_size,
        shrinking: configuration.shrinking,
    };
    let target_values: Vec<F> = responses.iter().copied().collect();
    let solved = sk_solve_epsilon_svr(
        &target_values,
        configuration.regularization_c,
        configuration.epsilon,
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
        backend: configuration.backend,
        builder_state: configuration.builder_state,
    })
}
