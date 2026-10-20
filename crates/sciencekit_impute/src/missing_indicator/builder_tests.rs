//! Companion tests for the missing-indicator builder (spec `missing-indicator`).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKUnsupervisedFit};

use super::builder::SKMissingIndicatorBuilder;
use super::core_implementation::SKMissingIndicator;
use super::scope::SKMissingFeatureScope;

#[test]
fn defaults_mask_only_the_features_missing_at_fit() {
    let estimator = SKMissingIndicator::<f64>::build().build().unwrap();
    assert_eq!(
        estimator.features(),
        SKMissingFeatureScope::MissingOnly,
        "the default scope is MissingOnly"
    );
    assert!(
        estimator.error_on_new(),
        "new missingness is rejected by default"
    );
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert!(!estimator.sparse_output());

    let model = estimator.fit(&array![[1.0, f64::NAN], [3.0, 4.0]]).unwrap();
    assert_eq!(model.scope(), &[1]);
    assert_eq!(model.number_of_input_columns(), 2);
    assert_eq!(model.number_of_output_columns(), 1);
    assert_eq!(model.backend(), SKBackendKind::Faer);
}

#[test]
fn the_builder_chains_every_option() {
    let estimator = SKMissingIndicator::<f64>::build()
        .features(SKMissingFeatureScope::All)
        .error_on_new(false)
        .sparse_output(true)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    assert_eq!(estimator.features(), SKMissingFeatureScope::All);
    assert!(!estimator.error_on_new());
    assert!(estimator.sparse_output());
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);

    let defaulted = SKMissingIndicatorBuilder::<f64>::new().build().unwrap();
    assert_eq!(defaulted.features(), SKMissingFeatureScope::MissingOnly);
}
