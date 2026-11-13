//! The `SKSupervisedFit` implementation of the neighbor classifier
//! (spec `nearest-neighbor-estimators`).

use ndarray::Array1;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};

use super::SKKNeighborsClassifier;
use super::model::SKKNeighborsClassifierModel;
use super::targets::sk_canonicalize_classes;
use crate::neighbors::{sk_prepare_search, sk_view_dimensions};

impl<F: SKFloat> SKSupervisedFit<F> for SKKNeighborsClassifier<F> {
    type Model = SKKNeighborsClassifierModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let targets = targets.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || sk_fit_classifier(self, features, &targets))
    }
}

/// Canonicalize the targets, prepare the search and assemble the model.
fn sk_fit_classifier<F: SKFloat>(
    estimator: &SKKNeighborsClassifier<F>,
    features: SKDataView<'_, F>,
    targets: &SKTargetView<'_, F>,
) -> Result<SKKNeighborsClassifierModel<F>, SKError> {
    let (class_indices, classes) = sk_canonicalize_classes(targets)?;
    let search = sk_prepare_search(features, &estimator.configuration)?;
    if class_indices.len() != search.training_rows() {
        return Err(SKError::shape_mismatch_2d(
            search.training_rows(),
            1,
            class_indices.len(),
            1,
        ));
    }
    Ok(SKKNeighborsClassifierModel {
        search,
        configuration: estimator.configuration,
        classes,
        class_indices: Array1::from(class_indices),
    })
}
