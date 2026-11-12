//! `SKSupervisedFit` implementation for [`SKLinearSVC`]
//! (spec `linear-support-vector-classifier`).

use ndarray::Array1;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};

use super::config::SKLinearSVCConfiguration;
use super::core_implementation::SKLinearSVC;
use super::design::{SKLinearSvcDesign, sk_linear_svc_design_access};
use super::model::{SKLinearSVCModel, SKLinearSvcHead};
use super::solver::sk_solve_linear_svc_dual;
use super::validation::{
    sk_canonicalize_linear_svc_targets, sk_validate_linear_svc_training_features,
};
use crate::kernel::sk_svm_view_dimensions;
use crate::svc::sk_resolve_class_weights;

impl<F: SKFloat> SKSupervisedFit<F> for SKLinearSVC<F> {
    type Model = SKLinearSVCModel<F>;
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
        sk_run_operation(attributes, || sk_fit_linear_svc(self, features, &targets))
    }
}

/// Fit one binary head from signed labels and per-row class weights.
pub(crate) fn sk_fit_linear_svc_head<F: SKFloat>(
    design: &SKLinearSvcDesign<'_, F>,
    labels: &[F],
    row_weights: &[F],
    configuration: &SKLinearSVCConfiguration<F>,
    maximum_number_of_iterations: usize,
) -> Result<SKLinearSvcHead<F>, SKError> {
    let access = sk_linear_svc_design_access(design);
    let outcome = sk_solve_linear_svc_dual(
        access,
        labels,
        row_weights,
        configuration.loss,
        configuration.regularization_c,
        configuration.tolerance,
        maximum_number_of_iterations,
    )?;
    let columns = access.number_of_columns();
    let intercept = if configuration.include_intercept {
        outcome.coefficients[columns]
    } else {
        F::zero()
    };
    Ok(SKLinearSvcHead {
        coefficients: Array1::from_vec(outcome.coefficients[..columns].to_vec()),
        intercept,
        iterations: outcome.iterations,
    })
}

fn sk_fit_linear_svc<F: SKFloat>(
    estimator: &SKLinearSVC<F>,
    features: SKDataView<'_, F>,
    targets: &SKTargetView<'_, F>,
) -> Result<SKLinearSVCModel<F>, SKError> {
    let configuration = &estimator.configuration;
    let (rows, columns) = sk_validate_linear_svc_training_features(features, "fit")?;
    let (class_indices, classes) = sk_canonicalize_linear_svc_targets(targets, rows)?;
    let number_of_classes = classes.number_of_classes();
    let per_class_weights =
        sk_resolve_class_weights(&configuration.class_weight, &class_indices, &classes)?;
    let design = SKLinearSvcDesign::dispatch(features, configuration.include_intercept)?;

    let row_weights: Vec<F> = class_indices
        .iter()
        .map(|&class_index| per_class_weights[class_index])
        .collect();

    if number_of_classes > 2 {
        return Err(SKError::UnsupportedOperation {
            operation: "fit",
            reason: "one-vs-rest multiclass fitting lands in the next task".to_owned(),
        });
    }
    let signed_labels: Vec<F> = class_indices
        .iter()
        .map(|&class_index| {
            if class_index == 1 {
                F::one()
            } else {
                -F::one()
            }
        })
        .collect();
    let head = sk_fit_linear_svc_head(
        &design,
        &signed_labels,
        &row_weights,
        configuration,
        configuration.maximum_number_of_iterations,
    )?;
    Ok(SKLinearSVCModel {
        classes,
        heads: vec![head],
        loss: configuration.loss,
        regularization_c: configuration.regularization_c,
        number_of_features: columns,
        include_intercept: configuration.include_intercept,
        backend: configuration.backend,
        builder_state: configuration.builder_state,
    })
}
