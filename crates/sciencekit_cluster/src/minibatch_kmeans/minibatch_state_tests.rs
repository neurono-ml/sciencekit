//! Companion tests for MiniBatchKMeans configuration, internal state and model
//! accessors (spec `mini-batch-k-means`, task 1.1).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKClusterLabel, SKExecutionMode};

use super::model::SKMiniBatchKMeansModel;
use super::state::SKMiniBatchState;
use super::{SKKMeansInitialization, SKMiniBatchKMeans};

#[test]
fn minibatch_state_initializes_zero_counts_and_builds_model_accessors() {
    let estimator = SKMiniBatchKMeans::<f64>::build().build().unwrap();
    assert_eq!(estimator.number_of_clusters(), 8);
    assert_eq!(
        estimator.initialization(),
        SKKMeansInitialization::KMeansPlusPlus
    );
    assert_eq!(estimator.number_of_initializations(), 3);
    assert_eq!(estimator.batch_size(), 1024);
    assert_eq!(estimator.initialization_batch_size(), 3072);
    assert_eq!(estimator.maximum_number_of_iterations(), 100);
    assert_eq!(estimator.maximum_number_of_epochs(), 1);
    assert_eq!(estimator.tolerance(), 0.0);
    assert!((estimator.reassignment_ratio() - 0.01).abs() < 1e-12);
    assert_eq!(estimator.random_seed(), 0);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);

    let centers = array![[0.0_f64, 1.0], [10.0, 11.0]];
    let state = SKMiniBatchState::new(centers.clone(), estimator.configuration);
    assert_eq!(state.counts.len(), 2);
    assert!(state.counts.iter().all(|&count| count == 0));
    assert_eq!(state.batches_processed, 0);
    assert_eq!(state.last_batch_inertia, 0.0);
    assert!(!state.converged);

    let labels = array![SKClusterLabel::cluster(0), SKClusterLabel::cluster(1)];
    let model = SKMiniBatchKMeansModel::from_state(state, Some(labels.clone()), 2.5, 1);
    assert_eq!(model.cluster_centers(), centers.view());
    assert_eq!(model.counts().len(), 2);
    assert_eq!(model.labels(), Some(labels.view()));
    assert!((model.inertia() - 2.5).abs() < 1e-12);
    assert_eq!(model.batches_processed(), 0);
    assert_eq!(model.epochs_consumed(), 1);
    assert!(!model.converged());
    assert_eq!(model.number_of_clusters(), 2);
    assert_eq!(model.number_of_features(), 2);
    assert_eq!(model.backend(), SKBackendKind::Faer);
    assert_eq!(model.execution_mode(), SKExecutionMode::Automatic);
}
