//! Tiny-data, scalar-genericity (`f32`/`f64`) and model export acceptance
//! scenarios for successive halving.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKDataView, SKError, SKExportable, SKFloat, SKImportable, SKRegressorPredictor,
    SKSupervisedFit, SKTargetView, SKTensorDataType,
};

use super::super::fixtures::{cyclic_labels, dense_design};
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::halving::{
    SKHalvingGridSearchClassifier, SKHalvingGridSearchClassifierModel,
    SKHalvingGridSearchRegressor, SKHalvingGridSearchRegressorModel,
};
use crate::split::kfold::SKKFold;

#[derive(Debug, Clone)]
struct SKGenericConstantRegressor<F: SKFloat> {
    constant: F,
}

struct SKGenericConstantRegressorModel<F: SKFloat> {
    constant: F,
}

impl<F: SKFloat> SKSupervisedFit<F> for SKGenericConstantRegressor<F> {
    type Model = SKGenericConstantRegressorModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let _ = (features.try_into()?, targets.try_into()?);
        Ok(SKGenericConstantRegressorModel {
            constant: self.constant,
        })
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKGenericConstantRegressorModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(view) => view.nrows(),
            SKDataView::Sparse(view) => view.rows(),
            _ => 0,
        };
        Ok(Array1::from_elem(rows, self.constant))
    }
}

#[test]
fn halving_acceptance_tiny_classifier_and_regressor_select_refitted_and_exportable_winners() {
    let features = dense_design(8, 2);
    let labels = cyclic_labels(8, 2);
    let classifier = SKHalvingGridSearchClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(0.0),
            SKMockThresholdClassifier::new(4.0),
            SKMockThresholdClassifier::new(8.0),
        ])
        .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
        .factor(2)
        .minimum_resource(4)
        .build()
        .unwrap();
    let clf_model = classifier.fit(features.view(), labels.view()).unwrap();
    assert!(clf_model.is_refitted());
    assert_eq!(clf_model.resource_schedule(), vec![4, 8]);
    assert_eq!(clf_model.candidate_count_trajectory(), vec![3, 2, 1]);
    assert_eq!(clf_model.best_model().unwrap().training_rows(), 8);

    let exported_clf = clf_model.export_state().unwrap();
    assert_eq!(
        exported_clf.metadata().estimator_name(),
        "mock.threshold_classifier"
    );
    assert_eq!(
        exported_clf.metadata().scalar_data_type(),
        SKTensorDataType::F64
    );
    assert!(matches!(
        SKHalvingGridSearchClassifierModel::<f64, SKMockThresholdClassifier>::import_state(
            &exported_clf
        ),
        Err(SKError::UnsupportedOperation {
            operation: "import_state",
            ..
        })
    ));

    let targets = Array1::from_iter((0..8).map(|row| row as f64));
    let regressor = SKHalvingGridSearchRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(3.5),
        ])
        .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
        .factor(2)
        .minimum_resource(4)
        .build()
        .unwrap();
    let reg_model = regressor.fit(features.view(), targets.view()).unwrap();
    assert_eq!(reg_model.best_candidate_index(), 1);
    assert_eq!(reg_model.best_model().unwrap().training_rows(), 8);
    assert_eq!(
        reg_model
            .export_state()
            .unwrap()
            .metadata()
            .estimator_name(),
        "mock.constant_regressor"
    );
    assert!(matches!(
        SKHalvingGridSearchRegressorModel::<f64, SKMockConstantRegressor>::import_state(
            &reg_model.export_state().unwrap()
        ),
        Err(SKError::UnsupportedOperation {
            operation: "import_state",
            ..
        })
    ));

    let unrefitted = SKHalvingGridSearchRegressor::build()
        .candidates(vec![SKMockConstantRegressor::new(1.0)])
        .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
        .minimum_resource(4)
        .refit(false)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(matches!(
        unrefitted.export_state().unwrap_err(),
        SKError::UnsupportedOperation {
            operation: "export_state",
            ..
        }
    ));
}

#[test]
fn halving_acceptance_f32_scalar_pipeline_runs_without_f64_detour() {
    let features = Array2::<f32>::zeros((12, 2));
    let targets = Array1::<f32>::from_elem(12, 2.5_f32);
    let estimator = SKHalvingGridSearchRegressor::<f32, _>::build()
        .candidates(vec![
            SKGenericConstantRegressor { constant: 0.0_f32 },
            SKGenericConstantRegressor { constant: 2.5_f32 },
            SKGenericConstantRegressor { constant: 5.0_f32 },
        ])
        .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
        .factor(2)
        .minimum_resource(6)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.best_candidate_index(), 1);
    assert_eq!(model.best_score(), 0.0_f32);
    assert_eq!(model.predict(features.view()).unwrap()[0], 2.5_f32);
}
