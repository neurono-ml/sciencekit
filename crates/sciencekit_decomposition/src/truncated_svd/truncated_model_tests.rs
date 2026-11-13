//! Companion tests for `SKTruncatedSVDAlgorithm` and `SKTruncatedSVDModel` state.

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilderState, SKExecutionMode};

use super::config::SKTruncatedSVDAlgorithm;
use super::model::SKTruncatedSVDModel;

#[test]
fn truncated_model_exposes_uncentered_factorization_parts_and_default_algorithm() {
    assert_eq!(
        SKTruncatedSVDAlgorithm::default(),
        SKTruncatedSVDAlgorithm::Randomized
    );
    assert_ne!(
        SKTruncatedSVDAlgorithm::Randomized,
        SKTruncatedSVDAlgorithm::Arpack
    );

    let mut builder_state = SKBuilderState::new();
    builder_state.execution_mode(SKExecutionMode::InProcessSynchronous);
    let model = SKTruncatedSVDModel::from_parts(
        array![[0.6_f64, 0.8, 0.0], [-0.8, 0.6, 0.0]],
        array![5.0_f64, 2.0],
        array![4.0_f64, 0.5],
        array![0.8_f64, 0.1],
        SKTruncatedSVDAlgorithm::Randomized,
        SKBackendKind::MatrixMultiply,
        builder_state,
    );

    assert_eq!(model.number_of_components(), 2);
    assert_eq!(model.number_of_features(), 3);
    assert_eq!(model.components().dim(), (2, 3));
    assert_eq!(model.singular_values(), array![5.0, 2.0].view());
    assert_eq!(model.explained_variance(), array![4.0, 0.5].view());
    assert_eq!(model.explained_variance_ratio(), array![0.8, 0.1].view());
    assert_eq!(model.algorithm(), SKTruncatedSVDAlgorithm::Randomized);
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}
