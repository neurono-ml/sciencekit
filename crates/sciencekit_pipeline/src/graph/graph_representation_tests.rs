//! Dense, sparse CSR and mixed representation flow, no-implicit-densification
//! and node-attributed representation rejection tests for
//! [`super::SKPipelineGraph`] (spec `pipeline-graph`, Task 3.1).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKError, SKFeatureBlock};
use sciencekit_impute::missing_indicator::{SKMissingFeatureScope, SKMissingIndicator};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sprs::{CsMat, TriMat};

use crate::SKPipelineTransformer;

use super::{SKNodeIdentifier, SKPipelineGraph, SKPipelineGraphModel};

fn dense_features() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0],
    )
    .unwrap()
}

fn sparse_csr_features() -> CsMat<f64> {
    let mut triplets = TriMat::new((6, 2));
    [
        (0, 1, 1.0),
        (1, 0, 2.0),
        (2, 1, 3.0),
        (3, 0, 4.0),
        (4, 1, 5.0),
        (5, 0, 6.0),
    ]
    .into_iter()
    .for_each(|(row, column, value)| triplets.add_triplet(row, column, value));
    triplets.to_csr()
}

fn node_is_sparse(model: &SKPipelineGraphModel<f64>, node: SKNodeIdentifier) -> bool {
    model.fitted_node_block(node).unwrap().is_sparse()
}

struct CscEmittingStage;

impl SKPipelineTransformer<f64> for CscEmittingStage {
    type Fitted = ();
    fn fit_stage(&self, _block: &SKFeatureBlock<f64>) -> Result<(), SKError> {
        Ok(())
    }
    fn transform_stage(
        &self,
        _fitted: &(),
        block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        let mut triplets = TriMat::new((block.number_of_rows(), 2));
        (0..block.number_of_rows()).for_each(|row| triplets.add_triplet(row, row % 2, 1.0));
        Ok(SKFeatureBlock::Sparse(triplets.to_csc()))
    }
}

#[test]
fn sparse_csr_diamond_flows_without_implicit_densification_and_matches_dense_predictions() {
    let sparse = sparse_csr_features();
    let dense = dense_features();
    let labels = Array1::from_vec(vec![0_i64, 1, 0, 1, 0, 1]);

    let build_diamond = || {
        let mut builder = SKPipelineGraph::start::<f64>();
        let root = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
        let left = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
        let right = builder.add_transformer(
            SKMissingIndicator::<f64>::build()
                .features(SKMissingFeatureScope::All)
                .sparse_output(true)
                .error_on_new(false)
                .build()
                .unwrap(),
        );
        let merge = builder.add_merge();
        let sink = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
        builder
            .connect(root, left)
            .connect(root, right)
            .connect(left, merge)
            .connect(right, merge)
            .connect(merge, sink);
        (builder.build().unwrap(), [root, left, right, merge, sink])
    };

    let (sparse_graph, nodes) = build_diamond();
    let fitted_sparse = sparse_graph.fit(sparse.view(), labels.view()).unwrap();
    nodes
        .iter()
        .for_each(|&node| assert!(node_is_sparse(&fitted_sparse, node)));
    let final_query_block = fitted_sparse.transform_to_final(sparse.view()).unwrap();
    assert!(final_query_block.is_sparse() && final_query_block.number_of_columns() == 4);

    let (dense_graph, dense_nodes) = build_diamond();
    let fitted_dense = dense_graph.fit(&dense, labels.view()).unwrap();
    assert!(!node_is_sparse(&fitted_dense, dense_nodes[0]));
    assert!(!node_is_sparse(&fitted_dense, dense_nodes[1]));
    assert!(node_is_sparse(&fitted_dense, dense_nodes[2]));
    assert!(node_is_sparse(&fitted_dense, dense_nodes[3]));

    assert_eq!(
        fitted_sparse.predict_labels(sparse.view()).unwrap(),
        fitted_dense.predict_labels(&dense).unwrap()
    );
    fitted_sparse
        .predict_probabilities(sparse.view())
        .unwrap()
        .iter()
        .zip(fitted_dense.predict_probabilities(&dense).unwrap().iter())
        .for_each(|(&found, &expected)| assert!((found - expected).abs() < 1e-12));
}

#[test]
fn representation_rejection_surfaces_failing_node_identifier_and_representation_name() {
    let sparse = sparse_csr_features();
    let dense = dense_features();
    let labels = Array1::from_vec(vec![0_i64, 1, 0, 1, 0, 1]);
    let targets = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut builder = SKPipelineGraph::start::<f64>();
    let sparse_root = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
    let dense_only_branch = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let sink = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder
        .connect(sparse_root, dense_only_branch)
        .connect(dense_only_branch, sink);
    let graph = builder.build().unwrap();

    let fit_error = graph
        .fit_classifier_checked(sparse.view(), labels.view())
        .err()
        .unwrap();
    assert_eq!(fit_error.node_identifier(), Some(dense_only_branch));
    assert!(matches!(
        fit_error.error(),
        SKError::UnsupportedRepresentation {
            representation: "csr",
            ..
        }
    ));
    assert!(fit_error.to_string().contains("node 1") && fit_error.to_string().contains("csr"));

    let fitted = graph.fit_classifier(&dense, labels.view()).unwrap();
    let predict_error = fitted.predict_labels_checked(sparse.view()).err().unwrap();
    assert_eq!(predict_error.node_identifier(), Some(dense_only_branch));
    assert!(predict_error.to_string().contains("csr"));
    assert!(fitted.predict_probabilities_checked(sparse.view()).is_err());

    let mut regressor_builder = SKPipelineGraph::start::<f64>();
    let scaler_node = regressor_builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let regressor_sink =
        regressor_builder.add_regressor(SKLinearRegression::build().build().unwrap());
    regressor_builder.connect(scaler_node, regressor_sink);
    let regressor_graph = regressor_builder.build().unwrap();
    assert_eq!(
        regressor_graph
            .fit_regressor_checked(sparse.view(), targets.view())
            .err()
            .unwrap()
            .node_identifier(),
        Some(scaler_node)
    );
    let fitted_regressor = regressor_graph
        .fit_regressor(&dense, targets.view())
        .unwrap();
    assert_eq!(
        fitted_regressor
            .predict_checked(sparse.view())
            .err()
            .unwrap()
            .node_identifier(),
        Some(scaler_node)
    );

    let mut csc_builder = SKPipelineGraph::start::<f64>();
    let csc_node = csc_builder.add_transformer(CscEmittingStage);
    let csc_sink = csc_builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    csc_builder.connect(csc_node, csc_sink);
    let csc_error = csc_builder
        .build()
        .unwrap()
        .fit_checked(&dense, labels.view())
        .err()
        .unwrap();
    assert_eq!(csc_error.node_identifier(), Some(csc_node));
    assert!(matches!(
        csc_error.error(),
        SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        }
    ));
}
