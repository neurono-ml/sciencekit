//! Classifier fit tests (task 4.2).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKFloat, SKSupervisedFit};

use crate::node::SKTreeNode;

use super::SKDecisionTreeClassifier;

/// A perfectly separable fixture: feature 0 separates classes 0 and 1.
fn separable() -> (ndarray::Array2<f64>, ndarray::Array1<i64>) {
    (array![[0.0], [1.0], [2.0], [3.0]], array![0_i64, 0, 1, 1])
}

/// The root split is placed between the two classes and both leaves are pure.
#[test]
fn perfectly_separable_fixture_yields_pure_leaves() {
    let (features, labels) = separable();
    let model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(features.view(), labels.view())
        .expect("fit succeeds");
    assert_eq!(model.node_count(), 3);
    let root = model.tree().node(0).expect("root exists");
    assert_eq!(root.split_feature(), Some(0));
    assert_eq!(root.threshold(), Some(1.5));
    assert_eq!(model.tree().node(1).expect("left").value(), &[1.0, 0.0]);
    assert_eq!(model.tree().node(2).expect("right").value(), &[0.0, 1.0]);
}

/// A multi-class fixture produces one class column per class and pure leaves.
#[test]
fn multi_class_fixture_builds_the_canonical_table() {
    let features = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let labels = array![10_i64, 10, 20, 20, 30, 30];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(features.view(), labels.view())
        .expect("fit succeeds");
    assert_eq!(model.number_of_classes(), 3);
    assert_eq!(model.classes().label_of(0), Some("10"));
    assert_eq!(model.classes().label_of(2), Some("30"));
    assert_eq!(model.node_count(), 5);
    let leaves: Vec<&SKTreeNode<f64>> = model
        .tree()
        .nodes()
        .iter()
        .filter(|node| node.is_leaf())
        .collect();
    assert_eq!(leaves.len(), 3);
    for leaf in leaves {
        assert_eq!(leaf.value().len(), 3);
        assert_eq!(leaf.value().iter().filter(|value| **value > 0.0).count(), 1);
    }
}

/// Nominal targets are canonicalized in first-occurrence order.
#[test]
fn nominal_targets_are_canonicalized() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let labels = ["beta", "beta", "alpha", "alpha"];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, &labels[..])
        .expect("fit succeeds");
    assert_eq!(model.classes().label_of(0), Some("beta"));
    assert_eq!(model.classes().label_of(1), Some("alpha"));
}

/// The configured backend and execution intent reach the fitted model.
#[test]
fn backend_and_execution_intent_reach_the_model() {
    let (features, labels) = separable();
    let model = SKDecisionTreeClassifier::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .expect("builds")
        .fit(features.view(), labels.view())
        .expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(model.feature_count(), 1);
}

/// Both supported floats fit through the same generic surface.
#[test]
fn fit_accepts_either_float() {
    fn nodes<F: SKFloat>() -> usize {
        let values: Vec<F> = [0.0, 1.0, 2.0, 3.0]
            .iter()
            .map(|value| F::from(*value).expect("value fits"))
            .collect();
        let features = ndarray::Array2::from_shape_vec((4, 1), values).expect("shape");
        let labels = ndarray::Array1::from_vec(vec![0_i64, 0, 1, 1]);
        SKDecisionTreeClassifier::<F>::build()
            .build()
            .expect("builds")
            .fit(features.view(), labels.view())
            .expect("fit succeeds")
            .node_count()
    }
    assert_eq!(nodes::<f32>(), nodes::<f64>());
}
