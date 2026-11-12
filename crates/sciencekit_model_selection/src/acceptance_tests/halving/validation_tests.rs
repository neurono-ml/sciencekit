//! Fit-time input, finiteness and dataset-relative resource-bound validation
//! acceptance scenarios for successive halving.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKError, SKSupervisedFit};
use sprs::TriMat;

use super::super::fixtures::dense_design;
use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::halving::SKHalvingGridSearchRegressor;
use crate::split::kfold::SKKFold;

#[test]
fn halving_acceptance_rejects_empty_mismatched_non_finite_and_oversized_resource_inputs() {
    let base = || {
        SKHalvingGridSearchRegressor::build()
            .candidates(vec![
                SKMockConstantRegressor::new(0.0),
                SKMockConstantRegressor::new(1.0),
            ])
            .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
    };

    let empty_features = Array2::<f64>::zeros((0, 2));
    let empty_targets = Array1::<f64>::zeros(0);
    assert!(matches!(
        base()
            .minimum_resource(2)
            .build()
            .unwrap()
            .fit(empty_features.view(), empty_targets.view()),
        Err(SKError::EmptyInput {
            operation: "halving_grid_search"
        })
    ));

    let features = dense_design(10, 2);
    let short_targets = Array1::<f64>::zeros(7);
    assert!(matches!(
        base()
            .minimum_resource(4)
            .build()
            .unwrap()
            .fit(features.view(), short_targets.view()),
        Err(SKError::ShapeMismatch { .. })
    ));

    let mut nan_targets = Array1::<f64>::zeros(10);
    nan_targets[9] = f64::INFINITY;
    assert!(matches!(
        base()
            .minimum_resource(4)
            .build()
            .unwrap()
            .fit(features.view(), nan_targets.view()),
        Err(SKError::NonFiniteValue { position: 9 })
    ));

    let mut triplets = TriMat::new((10, 2));
    triplets.add_triplet(8, 1, f64::NAN);
    let sparse_nan = triplets.to_csr();
    let clean_targets = Array1::<f64>::zeros(10);
    assert!(matches!(
        base()
            .minimum_resource(4)
            .build()
            .unwrap()
            .fit(sparse_nan.view(), clean_targets.view()),
        Err(SKError::NonFiniteValue { position: 17 })
    ));

    assert!(matches!(
        base()
            .minimum_resource(4)
            .maximum_resource(25)
            .build()
            .unwrap()
            .fit(features.view(), clean_targets.view()),
        Err(SKError::InvalidHyperparameter {
            name: "maximum_resource",
            ..
        })
    ));
    assert!(matches!(
        base()
            .minimum_resource(12)
            .build()
            .unwrap()
            .fit(features.view(), clean_targets.view()),
        Err(SKError::InvalidHyperparameter {
            name: "minimum_resource",
            ..
        })
    ));
}
