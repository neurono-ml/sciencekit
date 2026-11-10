//! Representation-flow tests (spec `typed-pipeline`).
//!
//! A dense or sparse block keeps its representation through the chain unless a
//! stage explicitly changes it, and the pipeline never densifies on its own.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKError, SKFeatureBlock, SKFeatureTransformer,
    SKUnsupervisedFit,
};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sprs::{CsMat, TriMat};

use crate::{SKConfiguredStages, SKPipeline};

fn labels() -> Array1<i64> {
    Array1::from_vec(vec![0, 1, 0, 1, 0, 1])
}

fn dense_features() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0],
    )
    .unwrap()
}

fn sparse_features() -> CsMat<f64> {
    let mut triplets = TriMat::new((6, 2));
    for (row, column, value) in [
        (0, 1, 1.0),
        (1, 0, 2.0),
        (2, 1, 3.0),
        (3, 0, 4.0),
        (4, 1, 5.0),
        (5, 0, 6.0),
    ] {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

fn imputer() -> SKSimpleImputer<f64> {
    SKSimpleImputer::build().build().unwrap()
}

fn classifier() -> SKLogisticRegression<f64> {
    SKLogisticRegression::build().build().unwrap()
}

type FittedChain = (((), SKSimpleImputer<f64>), SKSimpleImputer<f64>);
type FittedPipeline = crate::SKPipelineClassifier<
    f64,
    FittedChain,
    sciencekit_linear_model::logistic_regression::model::SKLogisticRegressionModel<f64>,
>;

fn imputer_chain() -> FittedPipeline {
    SKPipeline::start::<f64>()
        .then(imputer())
        .then(imputer())
        .finalize_classifier(classifier())
        .unwrap()
        .fit(sparse_features().view(), labels().view())
        .unwrap()
}

#[test]
fn a_sparse_chain_keeps_the_csr_representation() {
    let fitted = imputer_chain();
    let block = fitted
        .stages()
        .transform_stages(
            fitted.fitted_stages(),
            &SKFeatureBlock::Sparse(sparse_features()),
        )
        .unwrap();
    assert!(block.is_sparse());
    assert_eq!(block.number_of_columns(), 2);
}

#[test]
fn dense_input_stays_dense_through_the_chain() {
    let fitted = imputer_chain();
    let block = fitted
        .stages()
        .transform_stages(
            fitted.fitted_stages(),
            &SKFeatureBlock::from(dense_features()),
        )
        .unwrap();
    assert!(!block.is_sparse());
}

#[test]
fn sparse_and_dense_inputs_predict_the_same_labels() {
    let sparse_pipeline = imputer_chain();
    let dense_pipeline = SKPipeline::start::<f64>()
        .then(imputer())
        .then(imputer())
        .finalize_classifier(classifier())
        .unwrap()
        .fit(&dense_features(), labels().view())
        .unwrap();
    let sparse_labels = sparse_pipeline
        .predict_labels(sparse_features().view())
        .unwrap();
    let dense_labels = dense_pipeline.predict_labels(&dense_features()).unwrap();
    assert_eq!(sparse_labels, dense_labels);

    let sparse_probabilities = sparse_pipeline
        .predict_probabilities(sparse_features().view())
        .unwrap();
    let dense_probabilities = dense_pipeline
        .predict_probabilities(&dense_features())
        .unwrap();
    for (sparse, dense) in sparse_probabilities.iter().zip(dense_probabilities.iter()) {
        assert!((sparse - dense).abs() < 1e-12);
    }
}

/// A synthetic stage whose documented output is dense, used to prove a
/// representation change is the stage's choice and not a pipeline side effect.
#[derive(Clone)]
struct SKDensifyingStage;

struct SKDensifyingStageModel;

impl SKUnsupervisedFit<f64> for SKDensifyingStage {
    type Model = SKDensifyingStageModel;
    type Error = SKError;

    fn fit<'a, X>(&self, _features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        Ok(SKDensifyingStageModel)
    }
}

impl SKFeatureTransformer<f64> for SKDensifyingStageModel {
    type Output = Array2<f64>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        match features.try_into()? {
            SKDataView::Dense(view) => Ok(view.to_owned()),
            SKDataView::Sparse(view) => Ok(view.to_dense()),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the input to a dense or CSR matrix",
            }),
        }
    }
}

#[test]
fn a_stage_that_outputs_dense_changes_the_representation_explicitly() {
    let fitted = SKPipeline::start::<f64>()
        .then(imputer())
        .then(SKDensifyingStage)
        .finalize_classifier(classifier())
        .unwrap()
        .fit(sparse_features().view(), labels().view())
        .unwrap();
    let block = fitted
        .stages()
        .transform_stages(
            fitted.fitted_stages(),
            &SKFeatureBlock::Sparse(sparse_features()),
        )
        .unwrap();
    assert!(!block.is_sparse());
    assert!(fitted.predict_labels(sparse_features().view()).is_ok());
}

#[test]
fn the_block_type_exposes_the_flow_without_a_pipeline_conversion() {
    // A sparse block handed to the stage chain stays sparse even before the
    // final estimator, so no implicit densification happened at the boundary.
    let fitted = imputer_chain();
    let input = SKFeatureBlock::Sparse(sparse_features());
    let after_one = fitted
        .stages()
        .transform_stages(fitted.fitted_stages(), &input)
        .unwrap();
    assert!(input.is_sparse());
    assert!(after_one.is_sparse());
}
