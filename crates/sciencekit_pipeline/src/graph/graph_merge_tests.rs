//! Explicit merge nodes and edge-insertion-order horizontal concatenation tests
//! (spec `pipeline-graph`, Task 2.4).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKFeatureBlock};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::SKPipelineTransformer;

use super::SKPipelineGraph;

struct ConstantColumnStage {
    fill_value: f64,
    width: usize,
}

impl SKPipelineTransformer<f64> for ConstantColumnStage {
    type Fitted = (f64, usize);

    fn fit_stage(&self, _block: &SKFeatureBlock<f64>) -> Result<Self::Fitted, SKError> {
        Ok((self.fill_value, self.width))
    }

    fn transform_stage(
        &self,
        &(fill_value, width): &Self::Fitted,
        block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        let rows = block.number_of_rows();
        Ok(SKFeatureBlock::Dense(Array2::from_elem(
            (rows, width),
            fill_value,
        )))
    }
}

struct FixedRowStage {
    rows: usize,
}

impl SKPipelineTransformer<f64> for FixedRowStage {
    type Fitted = usize;

    fn fit_stage(&self, _block: &SKFeatureBlock<f64>) -> Result<Self::Fitted, SKError> {
        Ok(self.rows)
    }

    fn transform_stage(
        &self,
        &rows: &Self::Fitted,
        _block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        Ok(SKFeatureBlock::Dense(Array2::zeros((rows, 1))))
    }
}

#[test]
fn explicit_merge_node_concatenates_columns_in_exact_edge_insertion_order() {
    let features = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let labels = Array1::from_vec(vec![0_i64, 0, 1, 1]);
    let query = array![[9.0_f64, 10.0], [11.0, 12.0]];

    // Branch node identifiers are 0 (10.0, width 1), 1 (20.0, width 2), 2 (30.0, width 1),
    // connected into merge_node in order [node 2, node 0, node 1].
    let mut builder = SKPipelineGraph::start::<f64>();
    let branch_ten = builder.add_transformer(ConstantColumnStage {
        fill_value: 10.0,
        width: 1,
    });
    let branch_twenty = builder.add_transformer(ConstantColumnStage {
        fill_value: 20.0,
        width: 2,
    });
    let branch_thirty = builder.add_transformer(ConstantColumnStage {
        fill_value: 30.0,
        width: 1,
    });
    let merge_node = builder.add_merge();
    let final_node = builder.add_classifier(SKLogisticRegression::build().build().unwrap());

    builder
        .connect(branch_thirty, merge_node)
        .connect(branch_ten, merge_node)
        .connect(branch_twenty, merge_node)
        .connect(merge_node, final_node);

    assert_eq!(
        builder.incoming_nodes(merge_node),
        vec![branch_thirty, branch_ten, branch_twenty]
    );

    let graph = builder.build().unwrap();
    assert_eq!(
        graph.incoming_nodes(merge_node),
        vec![branch_thirty, branch_ten, branch_twenty]
    );

    let fitted = graph.fit(&features, labels.view()).unwrap();
    assert_eq!(fitted.incoming_widths(merge_node), Some(&[1, 1, 2][..]));

    let merged_fit = match fitted.fitted_node_block(merge_node).unwrap() {
        SKFeatureBlock::Dense(matrix) => matrix.clone(),
        other => panic!("expected dense merged block, got {other:?}"),
    };
    assert_eq!(merged_fit.row(0).to_vec(), vec![30.0_f64, 10.0, 20.0, 20.0]);

    let merged_query = match fitted.transform_to_final(&query).unwrap() {
        SKFeatureBlock::Dense(matrix) => matrix,
        other => panic!("expected dense transformed block, got {other:?}"),
    };
    assert_eq!(merged_query.nrows(), 2);
    assert_eq!(
        merged_query.row(1).to_vec(),
        vec![30.0_f64, 10.0, 20.0, 20.0]
    );
}

#[test]
fn merge_node_feeding_downstream_transformer_and_mismatch_attribution_work() {
    let features = array![
        [1.0_f64, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0],
    ];
    let targets = array![3.0_f64, 6.0, 9.0, 12.0, 15.0];

    let mut builder = SKPipelineGraph::start::<f64>();
    let branch_left = builder.add_transformer(ConstantColumnStage {
        fill_value: 2.0,
        width: 1,
    });
    let branch_right = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let merge_node = builder.add_merge();
    let post_merge_scaler = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let regressor_node = builder.add_regressor(SKLinearRegression::build().build().unwrap());

    builder
        .connect(branch_left, merge_node)
        .connect(branch_right, merge_node)
        .connect(merge_node, post_merge_scaler)
        .connect(post_merge_scaler, regressor_node);

    let fitted = builder
        .build()
        .unwrap()
        .fit(&features, targets.view())
        .unwrap();
    assert_eq!(fitted.number_of_features(), 3);
    assert_eq!(fitted.predict(&features).unwrap().len(), 5);

    // A branch emitting the wrong row count fails with node attribution.
    let mut broken = SKPipelineGraph::start::<f64>();
    let bad_branch = broken.add_transformer(FixedRowStage { rows: 2 });
    let good_branch = broken.add_transformer(SKStandardScaler::build().build().unwrap());
    let bad_merge = broken.add_merge();
    let sink = broken.add_regressor(SKLinearRegression::build().build().unwrap());
    broken
        .connect(good_branch, bad_merge)
        .connect(bad_branch, bad_merge)
        .connect(bad_merge, sink);

    let error = broken
        .build()
        .unwrap()
        .fit_checked(&features, targets.view())
        .err()
        .unwrap();
    assert_eq!(error.node_identifier(), Some(bad_branch));
    assert!(matches!(error.error(), SKError::ShapeMismatch { .. }));
}
