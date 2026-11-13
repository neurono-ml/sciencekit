//! `SKSupervisedFit` implementation for [`SKLinearSVC`]
//! (spec `linear-support-vector-classifier`).

use ndarray::Array1;
use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};

use super::config::SKLinearSVCConfiguration;
use super::core_implementation::SKLinearSVC;
use super::design::{SKLinearSvcDesign, sk_linear_svc_design_access};
use super::model::{SKLinearSVCHead, SKLinearSVCModel};
use super::solver::sk_solve_linear_svc_dual;
use super::validation::{
    sk_canonicalize_linear_svc_targets, sk_validate_linear_svc_training_features,
};
use crate::kernel::{sk_resolve_svm_execution_plan, sk_svm_view_dimensions};
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
) -> Result<SKLinearSVCHead<F>, SKError> {
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
    Ok(SKLinearSVCHead {
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
    let plan =
        sk_resolve_svm_execution_plan(&configuration.builder_state, &features, rows, columns)?;
    let (class_indices, classes) = sk_canonicalize_linear_svc_targets(targets, rows)?;
    let number_of_classes = classes.number_of_classes();
    let per_class_weights =
        sk_resolve_class_weights(&configuration.class_weight, &class_indices, &classes)?;
    let design = SKLinearSvcDesign::dispatch(features, configuration.include_intercept)?;

    let row_weights: Vec<F> = class_indices
        .iter()
        .map(|&class_index| per_class_weights[class_index])
        .collect();

    let head_labels: Vec<Vec<F>> = if number_of_classes == 2 {
        // Binary: one head, positive side on the larger canonical class.
        vec![
            class_indices
                .iter()
                .map(|&class_index| {
                    if class_index == 1 {
                        F::one()
                    } else {
                        -F::one()
                    }
                })
                .collect(),
        ]
    } else {
        // One-vs-rest: head `c` separates class `c` from the remaining classes.
        (0..number_of_classes)
            .map(|head_class| {
                class_indices
                    .iter()
                    .map(|&class_index| {
                        if class_index == head_class {
                            F::one()
                        } else {
                            -F::one()
                        }
                    })
                    .collect()
            })
            .collect()
    };
    let fit_head = |signed_labels: &Vec<F>| {
        sk_fit_linear_svc_head(
            &design,
            signed_labels,
            &row_weights,
            configuration,
            configuration.maximum_number_of_iterations,
        )
    };
    let heads = if plan.parallelism > 1 {
        head_labels
            .par_iter()
            .map(fit_head)
            .collect::<Result<Vec<_>, SKError>>()?
    } else {
        head_labels
            .iter()
            .map(fit_head)
            .collect::<Result<Vec<_>, SKError>>()?
    };
    Ok(SKLinearSVCModel {
        classes,
        heads,
        loss: configuration.loss,
        regularization_c: configuration.regularization_c,
        number_of_features: columns,
        include_intercept: configuration.include_intercept,
        backend: configuration.backend,
        builder_state: configuration.builder_state,
    })
}
