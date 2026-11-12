//! Pipeline integration tests: `SKFeatureUnion` inside `SKPipeline` and
//! sub-pipelines inside `SKFeatureUnion` (spec `feature-union`, Task 3.1).

use ndarray::{Array1, Array2, Axis, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKError, SKFeatureBlock, SKRegressorPredictor,
};
use sciencekit_impute::missing_indicator::SKMissingIndicator;
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::one_hot_encoder::{
    SKEncodedOutput, SKOneHotEncoder, SKOneHotEncoderModel,
};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sciencekit_preprocessing::standard_scaler::model::SKStandardScalerModel;

use crate::{SKConfiguredStages, SKFeatureUnion, SKPipeline, SKPipelineTransformer};

/// Selects column 0 (numeric) and applies `SKStandardScaler`.
struct SKNumericColumnScaler {
    scaler: SKStandardScaler,
}

impl SKPipelineTransformer<f64> for SKNumericColumnScaler {
    type Fitted = SKStandardScalerModel<f64>;

    fn fit_stage(&self, block: &SKFeatureBlock<f64>) -> Result<Self::Fitted, SKError> {
        let dense = block.view().as_dense()?;
        let numeric = dense.select(Axis(1), &[0]);
        use sciencekit_common::SKUnsupervisedFit;
        self.scaler.fit(numeric.view())
    }

    fn transform_stage(
        &self,
        fitted: &Self::Fitted,
        block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        let dense = block.view().as_dense()?;
        let numeric = dense.select(Axis(1), &[0]);
        use sciencekit_common::SKFeatureTransformer;
        Ok(SKFeatureBlock::Dense(fitted.transform(numeric.view())?))
    }
}

/// Extracts integer-coded column 1 and applies `SKOneHotEncoder`.
struct SKCategoricalColumnOneHot {
    encoder: SKOneHotEncoder,
}

impl SKCategoricalColumnOneHot {
    fn extract_codes(block: &SKFeatureBlock<f64>) -> Result<Array2<i64>, SKError> {
        let dense = block.view().as_dense()?;
        Ok(dense.select(Axis(1), &[1]).mapv(|value| value as i64))
    }
}

impl SKPipelineTransformer<f64> for SKCategoricalColumnOneHot {
    type Fitted = SKOneHotEncoderModel;

    fn fit_stage(&self, block: &SKFeatureBlock<f64>) -> Result<Self::Fitted, SKError> {
        let codes = Self::extract_codes(block)?;
        self.encoder.fit(codes.view())
    }

    fn transform_stage(
        &self,
        fitted: &Self::Fitted,
        block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        let codes = Self::extract_codes(block)?;
        match fitted.transform::<f64, _>(codes.view())? {
            SKEncodedOutput::Dense(matrix) => Ok(SKFeatureBlock::Dense(matrix)),
            SKEncodedOutput::Sparse(matrix) => Ok(SKFeatureBlock::Sparse(matrix)),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "use dense or CSR one-hot output",
            }),
        }
    }
}

fn mixed_table() -> (Array2<f64>, Array1<i64>) {
    let features = array![
        [1.0_f64, 0.0],
        [1.5, 0.0],
        [2.0, 1.0],
        [8.0, 1.0],
        [8.5, 2.0],
        [9.0, 2.0],
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    (features, labels)
}

#[test]
fn scaler_union_one_hot_encoder_into_classifier_concatenates_columns_and_predicts() {
    let (features, labels) = mixed_table();
    let union = SKFeatureUnion::build()
        .stages((
            SKNumericColumnScaler {
                scaler: SKStandardScaler::build().build().unwrap(),
            },
            SKCategoricalColumnOneHot {
                encoder: SKOneHotEncoder::build()
                    .sparse_output(true)
                    .build()
                    .unwrap(),
            },
        ))
        .build()
        .unwrap();

    let fitted_pipeline = SKPipeline::start::<f64>()
        .then(union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();

    assert_eq!(fitted_pipeline.number_of_stages(), 1);
    assert_eq!(fitted_pipeline.number_of_features(), 4);

    let (_, fitted_union) = fitted_pipeline.fitted_stages();
    assert_eq!(fitted_union.member_widths(), &[1, 3]);

    // Verify column order: scaler column 0 followed by 3 one-hot indicator columns, in CSR.
    let transformed = fitted_pipeline
        .stages()
        .transform_stages(
            fitted_pipeline.fitted_stages(),
            SKFeatureBlock::from(features.clone()),
        )
        .unwrap();
    assert!(transformed.is_sparse());
    match transformed.view() {
        SKDataView::Sparse(csr) => {
            let dense = csr.to_dense();
            // Row 0 belongs to category 0 -> indicator [1, 0, 0] in columns 1..4.
            assert_eq!(
                (dense[[0, 1]], dense[[0, 2]], dense[[0, 3]]),
                (1.0, 0.0, 0.0)
            );
            // Row 4 belongs to category 2 -> indicator [0, 0, 1] in columns 1..4.
            assert_eq!(
                (dense[[4, 1]], dense[[4, 2]], dense[[4, 3]]),
                (0.0, 0.0, 1.0)
            );
        }
        _ => panic!("expected CSR output from mixed scaler + sparse encoder"),
    }

    let predictions = fitted_pipeline.predict_labels(&features).unwrap();
    assert_eq!(predictions, labels);
}

#[test]
fn sub_pipeline_branch_composes_inside_feature_union_and_regressor_pipeline() {
    let features = array![[1.0_f64, f64::NAN], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0],];
    let targets = array![3.0_f64, 6.0, 9.0, 12.0];

    let impute_then_scale = SKPipeline::start::<f64>()
        .then(SKSimpleImputer::<f64>::build().build().unwrap())
        .then(SKStandardScaler::build().build().unwrap());
    let missing_mask = SKMissingIndicator::<f64>::build()
        .error_on_new(false)
        .build()
        .unwrap();

    let union = SKFeatureUnion::build()
        .stages((impute_then_scale, missing_mask))
        .build()
        .unwrap();

    let fitted = SKPipeline::start::<f64>()
        .then(union)
        .finalize_regressor(SKLinearRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, targets.view())
        .unwrap();

    assert_eq!(fitted.number_of_features(), 3);
    let responses = fitted.predict(&features).unwrap();
    assert_eq!(responses.len(), 4);
    assert!(responses.iter().all(|value| value.is_finite()));
}
