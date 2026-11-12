//! Companion tests for the structure-only build and leaf-value replacement
//! surface (tasks 1.1 and 1.2).

use ndarray::array;
use sciencekit_common::SKError;
use sprs::CsMat;

use super::{SKStructureConstraints, SKTreeStructure};

/// Growing a structure on pseudo-residuals partitions the rows without
/// committing to the public regressor's leaf-mean output, reports leaf count
/// and sample membership, and replaces leaf values while preserving splits.
#[test]
fn build_structure_and_replace_leaf_values_preserve_split_geometry() {
    let features = array![[1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
    let pseudo_residuals = array![-2.0, -1.0, 3.0, 5.0];
    let constraints = SKStructureConstraints::<f64>::build()
        .maximum_depth(1)
        .minimum_samples_leaf(2)
        .build()
        .expect("constraints build");

    let mut structure =
        SKTreeStructure::build_structure(&features, pseudo_residuals.view(), &constraints)
            .expect("structure builds");

    assert_eq!(structure.sample_count(), 4);
    assert_eq!(structure.leaf_count(), 2);
    assert_eq!(structure.tree().leaf_count(), 2);
    assert_eq!(structure.leaf_node_indices(), &[1, 2]);
    assert_eq!(structure.leaf_sample_indices(0), Some(&[0usize, 1][..]));
    assert_eq!(structure.leaf_sample_indices(1), Some(&[2usize, 3][..]));

    // Before replacement, structure-only leaves hold neutral zero placeholders
    // rather than the pseudo-residual means (-1.5 and 4.0).
    assert_eq!(structure.tree().node(1).expect("left leaf").value(), &[0.0]);
    assert_eq!(
        structure.tree().node(2).expect("right leaf").value(),
        &[0.0]
    );

    let root_before = structure.tree().node(0).expect("root").clone();
    let newton_values = [-0.75, 2.25];
    structure
        .replace_leaf_values(&newton_values)
        .expect("leaf replacement succeeds");

    let root_after = structure.tree().node(0).expect("root");
    assert_eq!(root_after.split_feature(), root_before.split_feature());
    assert_eq!(root_after.threshold(), root_before.threshold());
    assert_eq!(root_after.left_child(), root_before.left_child());
    assert_eq!(root_after.right_child(), root_before.right_child());
    assert_eq!(root_after.sample_count(), root_before.sample_count());

    let tree = structure.into_tree();
    let left_leaf = tree.leaf_index_for(|feature| if feature == 0 { 1.5 } else { 0.0 });
    let right_leaf = tree.leaf_index_for(|feature| if feature == 0 { 10.5 } else { 0.0 });
    assert_eq!(tree.node(left_leaf).expect("left").value(), &[-0.75]);
    assert_eq!(tree.node(right_leaf).expect("right").value(), &[2.25]);
}

/// Supplying the wrong number of leaf values returns `ShapeMismatch` and leaves
/// every node in the tree untouched.
#[test]
fn replace_leaf_values_validates_length_and_leaves_structure_unchanged() {
    let features = array![[0.0], [1.0], [8.0], [9.0]];
    let pseudo_residuals = array![-1.0, -1.0, 2.0, 2.0];
    let constraints = SKStructureConstraints::<f64>::build()
        .maximum_depth(1)
        .build()
        .expect("constraints build");

    let mut structure =
        SKTreeStructure::build_structure(&features, pseudo_residuals.view(), &constraints)
            .expect("structure builds");
    structure
        .replace_leaf_values(&[-0.5, 1.0])
        .expect("initial replacement succeeds");
    let snapshot = structure.tree().clone();

    let mismatch = structure.replace_leaf_values(&[3.0]);
    assert!(matches!(
        mismatch,
        Err(SKError::ShapeMismatch {
            ref expected,
            ref found,
        }) if expected == &[2] && found == &[1]
    ));
    assert_eq!(structure.tree(), &snapshot);
}

/// Borrowed CSR inputs build the identical split structure and leaf membership
/// without densifying.
#[test]
fn build_structure_supports_sparse_csr_without_densifying() {
    let dense = array![[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [6.0, 1.0]];
    let sparse = CsMat::new(
        (4, 2),
        vec![0, 0, 1, 2, 4],
        vec![1, 0, 0, 1],
        vec![1.0, 5.0, 6.0, 1.0],
    );
    let pseudo_residuals = array![-1.0, -1.5, 2.0, 2.5];
    let constraints = SKStructureConstraints::<f64>::build()
        .maximum_depth(2)
        .build()
        .expect("constraints build");

    let dense_structure =
        SKTreeStructure::build_structure(&dense, pseudo_residuals.view(), &constraints)
            .expect("dense structure");
    let sparse_structure =
        SKTreeStructure::build_structure(sparse.view(), pseudo_residuals.view(), &constraints)
            .expect("sparse structure");

    assert_eq!(dense_structure.tree(), sparse_structure.tree());
    assert_eq!(
        dense_structure.leaf_node_indices(),
        sparse_structure.leaf_node_indices()
    );
    assert_eq!(
        dense_structure.feature_importances(),
        sparse_structure.feature_importances()
    );
}

/// `SKDecisionTreeRegressor` fitted through its public builder uses its
/// configured criterion (`SquaredError` leaf mean and `AbsoluteError` leaf
/// median) and is unaffected by the structure-only surface.
#[test]
fn public_decision_tree_regressor_is_unaffected_by_structure_surface() {
    use crate::{SKDecisionTreeRegressor, SKTreeCriterion};
    use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};

    let features = array![[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]];
    let responses = array![1.0, 2.0, 9.0, 10.0, 20.0, 90.0];

    let squared_model = SKDecisionTreeRegressor::<f64>::build()
        .criterion(SKTreeCriterion::SquaredError)
        .maximum_depth(1)
        .minimum_samples_leaf(3)
        .build()
        .expect("squared regressor builds")
        .fit(&features, responses.view())
        .expect("squared regressor fits");
    let absolute_model = SKDecisionTreeRegressor::<f64>::build()
        .criterion(SKTreeCriterion::AbsoluteError)
        .maximum_depth(1)
        .minimum_samples_leaf(3)
        .build()
        .expect("absolute regressor builds")
        .fit(&features, responses.view())
        .expect("absolute regressor fits");

    let queries = array![[0.5], [11.5]];
    let squared_predictions = squared_model.predict(&queries).expect("squared predict");
    let absolute_predictions = absolute_model.predict(&queries).expect("absolute predict");
    // Left leaf [1, 2, 9]: mean = 4.0, median = 2.0.
    // Right leaf [10, 20, 90]: mean = 40.0, median = 20.0.
    assert!((squared_predictions[0] - 4.0).abs() < 1e-12);
    assert!((squared_predictions[1] - 40.0).abs() < 1e-12);
    assert!((absolute_predictions[0] - 2.0).abs() < 1e-12);
    assert!((absolute_predictions[1] - 20.0).abs() < 1e-12);

    let constraints = SKStructureConstraints::<f64>::build()
        .maximum_depth(1)
        .minimum_samples_leaf(3)
        .build()
        .expect("constraints build");
    let mut structure = SKTreeStructure::build_structure(&features, responses.view(), &constraints)
        .expect("structure builds");
    assert_eq!(structure.tree().leaf_value_for(|_| 0.5), 0.0);
    structure
        .replace_leaf_values(&[-1.25, 3.75])
        .expect("replace succeeds");
    assert_eq!(structure.tree().leaf_value_for(|_| 0.5), -1.25);
    assert_eq!(structure.tree().leaf_value_for(|_| 11.5), 3.75);

    // Re-evaluating the public models confirms no shared state or drift.
    assert_eq!(
        squared_model.predict(&queries).expect("squared predict"),
        squared_predictions
    );
}
