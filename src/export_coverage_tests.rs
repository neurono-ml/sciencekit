//! Workspace-wide export coverage and concurrent determinism.
//!
//! The compile-time assertions prove every retrofitted fitted-model type
//! implements the shared contract; the runtime checks prove the shared format
//! version and that concurrent exports agree.

use ndarray::array;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKBuilder, SKExportable, SKUnsupervisedFit, sk_states_equal,
};

fn assert_exportable<M: SKExportable<f64>>() {}

#[test]
fn every_fitted_model_type_implements_the_export_contract() {
    assert_exportable::<sciencekit_preprocessing::standard_scaler::model::SKStandardScalerModel<f64>>(
    );
    assert_exportable::<sciencekit_preprocessing::minmax_scaler::model::SKMinMaxScalerModel<f64>>();
    assert_exportable::<sciencekit_preprocessing::robust_scaler::model::SKRobustScalerModel<f64>>();
    assert_exportable::<sciencekit_preprocessing::one_hot_encoder::SKOneHotEncoderModel>();
    assert_exportable::<sciencekit_preprocessing::polynomial_features::SKPolynomialFeaturesModel>();
    assert_exportable::<sciencekit_impute::simple_imputer::SKSimpleImputerModel<f64>>();
    assert_exportable::<sciencekit_impute::missing_indicator::SKMissingIndicatorModel<f64>>();
    assert_exportable::<sciencekit_impute::knn::SKKNNImputerModel<f64>>();
    assert_exportable::<
        sciencekit_linear_model::linear_regression::model::SKLinearRegressionModel<f64>,
    >();
    assert_exportable::<sciencekit_linear_model::ridge::SKRidgeModel<f64>>();
    assert_exportable::<sciencekit_linear_model::lasso::SKLassoModel<f64>>();
    assert_exportable::<sciencekit_linear_model::elastic_net::SKElasticNetModel<f64>>();
    assert_exportable::<sciencekit_linear_model::logistic_regression::SKLogisticRegressionModel<f64>>(
    );
    assert_exportable::<sciencekit_linear_model::sgd_classifier::SKSGDClassifierModel<f64>>();
    assert_exportable::<sciencekit_linear_model::sgd_regressor::SKSGDRegressorModel<f64>>();
    assert_exportable::<sciencekit_neighbors::classifier::SKKNeighborsClassifierModel<f64>>();
    assert_exportable::<sciencekit_neighbors::regressor::SKKNeighborsRegressorModel<f64>>();
    assert_exportable::<sciencekit_svm::SKSVCModel<f64>>();
    assert_exportable::<sciencekit_svm::SKSVRModel<f64>>();
    assert_exportable::<sciencekit_svm::linear_svc::SKLinearSVCModel<f64>>();
    assert_exportable::<sciencekit_tree::SKDecisionTreeClassifierModel<f64>>();
    assert_exportable::<sciencekit_tree::SKDecisionTreeRegressorModel<f64>>();
    assert_exportable::<sciencekit_ensemble::random_forest::SKRandomForestClassifierModel<f64>>();
    assert_exportable::<sciencekit_ensemble::random_forest::SKRandomForestRegressorModel<f64>>();
    assert_exportable::<
        sciencekit_ensemble::gradient_boosting::SKGradientBoostingClassifierModel<f64>,
    >();
    assert_exportable::<
        sciencekit_ensemble::gradient_boosting::SKGradientBoostingRegressorModel<f64>,
    >();
    assert_exportable::<
        sciencekit_ensemble::histogram_boosting::SKHistGradientBoostingClassifierModel<f64>,
    >();
    assert_exportable::<
        sciencekit_ensemble::histogram_boosting::SKHistGradientBoostingRegressorModel<f64>,
    >();
    assert_exportable::<sciencekit_cluster::kmeans::SKKMeansModel<f64>>();
    assert_exportable::<sciencekit_cluster::minibatch_kmeans::SKMiniBatchKMeansModel<f64>>();
    assert_exportable::<sciencekit_cluster::dbscan::SKDBSCANModel<f64>>();
    assert_exportable::<sciencekit_decomposition::pca::SKPCAModel<f64>>();
    assert_exportable::<sciencekit_decomposition::truncated_svd::SKTruncatedSVDModel<f64>>();
    assert_exportable::<sciencekit_decomposition::incremental_pca::SKIncrementalPCAModel<f64>>();
    assert_exportable::<sciencekit_outlier::isolation_forest::SKIsolationForestModel<f64>>();
    assert_exportable::<
        sciencekit_pipeline::SKPipelineClassifier<
            f64,
            (),
            sciencekit_linear_model::logistic_regression::SKLogisticRegressionModel<f64>,
        >,
    >();
}

#[test]
fn every_exported_state_uses_the_shared_format_version() {
    let features = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 8.0]];
    let model =
        sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler::build()
            .build()
            .unwrap()
            .fit(features.view())
            .unwrap();
    let state = model.export_state().unwrap();
    assert_eq!(state.metadata().format_version(), SK_MODEL_FORMAT_VERSION);
    assert!(state.number_of_tensors() > 0);
}

#[test]
fn concurrent_export_agrees_with_the_single_threaded_state() {
    let features = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 8.0]];
    let model =
        sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler::build()
            .build()
            .unwrap()
            .fit(features.view())
            .unwrap();
    let expected = model.export_state().unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = model.clone();
            std::thread::spawn(move || model.export_state().unwrap())
        })
        .collect();
    for handle in handles {
        let exported = handle.join().unwrap();
        assert!(sk_states_equal(&expected, &exported));
    }
}
