//! Single-stage boosting update tests (task 4.1).

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sciencekit_tree::SKMaximumFeatures;

use crate::boosting::loss::{SKGradientLoss, SKSquaredErrorLoss};
use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::growth::SKHistogramGrowthParameters;
use crate::histogram_boosting::stage::{SKHistogramStageParameters, sk_fit_histogram_scalar_stage};

#[test]
fn hist_stage_applies_a_hand_computed_newton_update() {
    let dense = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("fixture");
    let dataset = SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 4).expect("build");
    let targets = [0.0, 0.0, 1.0, 1.0];
    let loss = SKSquaredErrorLoss::new();
    let initial_prediction = loss.initial_prediction(&targets);
    let mut raw_scores = vec![initial_prediction; 4];
    let parameters = SKHistogramStageParameters {
        growth: SKHistogramGrowthParameters {
            maximum_leaf_nodes: 3,
            minimum_samples_leaf: 1,
            l2_regularization: 0.0,
            maximum_features: SKMaximumFeatures::All,
            random_seed: 0,
        },
        learning_rate: 0.1,
        subsample: 1.0,
        random_seed: 0,
    };
    let tree = sk_fit_histogram_scalar_stage::<f64, _>(
        &loss,
        &dataset,
        &targets,
        &mut raw_scores,
        &parameters,
        0,
        &(0..4).collect::<Vec<usize>>(),
    )
    .expect("stage");
    assert_eq!(tree.tree.node_count(), 3);
    assert!((raw_scores[0] - 0.45).abs() < 1e-12);
    assert!((raw_scores[1] - 0.45).abs() < 1e-12);
    assert!((raw_scores[2] - 0.55).abs() < 1e-12);
    assert!((raw_scores[3] - 0.55).abs() < 1e-12);
}
