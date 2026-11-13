//! `SKSupervisedFit` implementation for `SKSVC`
//! (spec `support-vector-machines`).

use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};
use sciencekit_math::sk_resolve_kernel_parameters;

use super::binary_fit::sk_fit_binary_pair;
use super::core_implementation::SKSVC;
use super::model::SKSVCModel;
use super::targets::sk_canonicalize_svc_targets;
use super::weights::sk_resolve_class_weights;
use crate::kernel::{
    sk_resolve_svm_execution_plan, sk_svm_view_dimensions, sk_validate_svm_training_features,
};

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
        let (rows_count, columns_count) = sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || sk_fit_svc(self, features, &targets))
    }
}

fn sk_fit_svc<F: SKFloat>(
    estimator: &SKSVC<F>,
    features: SKDataView<'_, F>,
    targets: &SKTargetView<'_, F>,
) -> Result<SKSVCModel<F>, SKError> {
    let configuration = &estimator.configuration;
    let (rows, columns) = sk_validate_svm_training_features(features, configuration.kernel, "fit")?;
    let plan =
        sk_resolve_svm_execution_plan(&configuration.builder_state, &features, rows, columns)?;
    let (class_indices, classes) = sk_canonicalize_svc_targets(targets, rows)?;
    let number_of_classes = classes.number_of_classes();
    let per_class_weights =
        sk_resolve_class_weights(&configuration.class_weight, &class_indices, &classes)?;
    let kernel_parameters = sk_resolve_kernel_parameters(
        configuration.kernel,
        configuration.gamma,
        configuration.degree,
        configuration.coef0,
        features,
    )?;

    let pairs: Vec<(usize, usize)> = (0..number_of_classes)
        .flat_map(|class_a| {
            ((class_a + 1)..number_of_classes).map(move |class_b| (class_a, class_b))
        })
        .collect();

    let fit_pair = |(pair_index, (class_negative, class_positive)): (usize, (usize, usize))| {
        sk_fit_binary_pair(
            configuration,
            features,
            &class_indices,
            &per_class_weights,
            kernel_parameters,
            class_negative,
            class_positive,
            pair_index as u64,
        )
    };
    let pairwise_models = if plan.parallelism > 1 {
        pairs
            .into_par_iter()
            .enumerate()
            .map(fit_pair)
            .collect::<Result<Vec<_>, SKError>>()?
    } else {
        pairs
            .into_iter()
            .enumerate()
            .map(fit_pair)
            .collect::<Result<Vec<_>, SKError>>()?
    };

    Ok(SKSVCModel {
        classes,
        pairwise_models,
        kernel_parameters,
        number_of_features: columns,
        backend: configuration.backend,
        builder_state: configuration.builder_state,
        probability_enabled: configuration.probability,
    })
}
