//! Topological fitting tests over diamond and linear-equivalent DAGs (spec
//! `pipeline-graph`, Task 2.2).

use ndarray::{Array1, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKExecutionMode, SKExportable,
};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::linear_regression::model::SKLinearRegressionModel;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_linear_model::logistic_regression::model::SKLogisticRegressionModel;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::{SKFeatureUnion, SKPipeline};

use super::SKPipelineGraph;

#[test]
fn diamond_graph_fits_both_branches_and_matches_equivalent_linear_chain() {
    let features = array![
        [1.0_f64, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0],
        [7.0, 70.0],
        [8.0, 80.0],
    ];
    let labels = Array1::from_vec(vec![0_i64, 0, 0, 0, 1, 1, 1, 1]);

    let mut builder = SKPipelineGraph::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous);
    let imputer_node = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
    let standard_node = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let minmax_node = builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let classifier_node = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder
        .connect(imputer_node, standard_node)
        .connect(imputer_node, minmax_node)
        .connect(standard_node, classifier_node)
        .connect(minmax_node, classifier_node);

    let graph = builder.build().unwrap();
    let fitted_graph = graph.fit(&features, labels.view()).unwrap();

    assert_eq!(fitted_graph.node_count(), 4);
    assert_eq!(fitted_graph.edge_count(), 4);
    assert_eq!(
        fitted_graph.topological_order(),
        &[imputer_node, standard_node, minmax_node, classifier_node]
    );
    assert_eq!(fitted_graph.fitted_node_blocks().len(), 4);
    assert_eq!(fitted_graph.number_of_features(), 4);
    assert_eq!(fitted_graph.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        fitted_graph.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let standard_block = fitted_graph.fitted_node_block(standard_node).unwrap();
    let minmax_block = fitted_graph.fitted_node_block(minmax_node).unwrap();
    let merged_block = fitted_graph.fitted_node_block(classifier_node).unwrap();
    assert_eq!(standard_block.number_of_columns(), 2);
    assert_eq!(minmax_block.number_of_columns(), 2);
    assert_eq!(merged_block.number_of_columns(), 4);

    let linear_union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();
    let fitted_linear = SKPipeline::start::<f64>()
        .then(SKSimpleImputer::<f64>::build().build().unwrap())
        .then(linear_union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();

    let graph_final_model = fitted_graph
        .final_model()
        .downcast_reference::<SKLogisticRegressionModel<f64>>()
        .unwrap();
    let graph_labels = graph_final_model.predict_labels(merged_block).unwrap();
    let linear_labels = fitted_linear.predict_labels(&features).unwrap();
    assert_eq!(graph_labels, linear_labels);
}

#[test]
fn regressor_diamond_graph_learns_identical_weights_to_linear_feature_union_chain() {
    let features = array![
        [1.0_f64, 2.0],
        [2.0, 3.0],
        [3.0, 5.0],
        [4.0, 8.0],
        [5.0, 13.0],
        [6.0, 21.0],
    ];
    let targets = array![5.0_f64, 8.0, 13.0, 20.0, 31.0, 48.0];

    let mut builder = SKPipelineGraph::start::<f64>();
    let left_branch = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let right_branch = builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let regressor_node = builder.add_regressor(SKLinearRegression::build().build().unwrap());
    builder
        .connect(left_branch, regressor_node)
        .connect(right_branch, regressor_node);

    let fitted_graph = builder
        .build()
        .unwrap()
        .fit(&features, targets.view())
        .unwrap();

    let linear_equivalent = SKPipeline::start::<f64>()
        .then(
            SKFeatureUnion::build()
                .stages((
                    SKStandardScaler::build().build().unwrap(),
                    SKMinMaxScaler::<f64>::build().build().unwrap(),
                ))
                .build()
                .unwrap(),
        )
        .finalize_regressor(SKLinearRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, targets.view())
        .unwrap();

    let graph_regressor = fitted_graph
        .final_model()
        .downcast_reference::<SKLinearRegressionModel<f64>>()
        .unwrap();
    assert_eq!(
        graph_regressor.coefficients(),
        linear_equivalent.final_model().coefficients()
    );
    assert!(
        (graph_regressor.intercept() - linear_equivalent.final_model().intercept()).abs() < 1e-12
    );
}

#[test]
fn graph_pipeline_export_returns_a_typed_unsupported_operation() {
    let features = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
    let labels = Array1::from_vec(vec![0_i64, 0, 1, 1]);
    let mut builder = SKPipelineGraph::<f64>::build();
    let standard_node = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let classifier_node = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder.connect(standard_node, classifier_node);
    let fitted = builder
        .build()
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();
    assert!(matches!(
        fitted.export_state(),
        Err(sciencekit_common::SKError::UnsupportedOperation {
            operation: "export_state",
            ..
        })
    ));
}
