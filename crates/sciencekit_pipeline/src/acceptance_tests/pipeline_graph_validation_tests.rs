//! Doomed structural fixtures and edge-width drift acceptance tests for
//! [`crate::SKPipelineGraph`] (spec `pipeline-graph`, Task 3.3).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKError, SKFeatureBlock};
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::{SKPipelineGraph, SKPipelineTransformer};

struct WidthDriftingStage;

impl SKPipelineTransformer<f64> for WidthDriftingStage {
    type Fitted = usize;

    fn fit_stage(&self, _block: &SKFeatureBlock<f64>) -> Result<Self::Fitted, SKError> {
        Ok(2)
    }

    fn transform_stage(
        &self,
        &fit_width: &Self::Fitted,
        block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        let rows = block.number_of_rows();
        let columns = if rows == 8 { fit_width } else { fit_width + 1 };
        Ok(SKFeatureBlock::Dense(Array2::from_elem(
            (rows, columns),
            1.0,
        )))
    }
}

#[test]
fn doomed_graph_fixtures_and_single_edge_width_drift_surface_structured_errors() {
    let mut cyclic = SKPipelineGraph::start::<f64>();
    let first = cyclic.add_transformer(SKStandardScaler::build().build().unwrap());
    let second = cyclic.add_transformer(SKStandardScaler::build().build().unwrap());
    let sink = cyclic.add_classifier(SKLogisticRegression::build().build().unwrap());
    cyclic
        .connect(first, second)
        .connect(second, first)
        .connect(second, sink);
    assert!(matches!(
        cyclic.build(),
        Err(SKError::InvalidHyperparameter { name: "edges", .. })
    ));

    let mut missing_final = SKPipelineGraph::start::<f64>();
    let only_stage = missing_final.add_transformer(SKStandardScaler::build().build().unwrap());
    let second_stage = missing_final.add_transformer(SKStandardScaler::build().build().unwrap());
    missing_final.connect(only_stage, second_stage);
    assert!(matches!(
        missing_final.build(),
        Err(SKError::InvalidHyperparameter {
            name: "final_estimator",
            ..
        })
    ));

    let mut disconnected = SKPipelineGraph::start::<f64>();
    let root = disconnected.add_transformer(SKStandardScaler::build().build().unwrap());
    let _orphan = disconnected.add_transformer(SKStandardScaler::build().build().unwrap());
    let final_node = disconnected.add_classifier(SKLogisticRegression::build().build().unwrap());
    disconnected.connect(root, final_node);
    assert!(matches!(
        disconnected.build(),
        Err(SKError::InvalidHyperparameter { name: "edges", .. })
    ));

    let features = Array2::from_elem((8, 2), 1.0_f64);
    let labels = Array1::from_vec(vec![0_i64, 0, 0, 0, 1, 1, 1, 1]);
    let unseen_query = Array2::from_elem((2, 2), 1.0_f64);

    let mut drifting_builder = SKPipelineGraph::start::<f64>();
    let drifting = drifting_builder.add_transformer(WidthDriftingStage);
    let passthrough_merge = drifting_builder.add_merge();
    let final_classifier =
        drifting_builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    drifting_builder
        .connect(drifting, passthrough_merge)
        .connect(passthrough_merge, final_classifier);
    let fitted = drifting_builder
        .build()
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();
    let error = fitted.transform_checked(&unseen_query).err().unwrap();
    assert_eq!(error.node_identifier(), Some(passthrough_merge));
    assert!(matches!(error.error(), SKError::ShapeMismatch { .. }));
}
