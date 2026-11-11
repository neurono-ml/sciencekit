//! Synthetic test stages shared by `union_validation_tests` (spec
//! `feature-union`).

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKError, SKFeatureBlock};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sprs::CsMat;

use crate::SKPipelineTransformer;

pub(super) fn scaler() -> SKStandardScaler {
    SKStandardScaler::build().build().unwrap()
}

pub(super) struct SKFailingStage {
    pub(super) fail_on_fit: bool,
    pub(super) fail_on_transform_rows: Option<usize>,
}

impl SKPipelineTransformer<f64> for SKFailingStage {
    type Fitted = ();

    fn fit_stage(&self, _block: &SKFeatureBlock<f64>) -> Result<(), SKError> {
        if self.fail_on_fit {
            return Err(SKError::UnsupportedOperation {
                operation: "fit_stage",
                reason: "synthetic fit failure".to_string(),
            });
        }
        Ok(())
    }

    fn transform_stage(
        &self,
        _fitted: &(),
        block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        if self.fail_on_transform_rows == Some(block.number_of_rows()) {
            return Err(SKError::Conversion("synthetic transform failure".into()));
        }
        Ok(block.clone())
    }
}

pub(super) struct SKFixedTwoRowStage;

impl SKPipelineTransformer<f64> for SKFixedTwoRowStage {
    type Fitted = ();

    fn fit_stage(&self, _block: &SKFeatureBlock<f64>) -> Result<(), SKError> {
        Ok(())
    }

    fn transform_stage(
        &self,
        _fitted: &(),
        _block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        Ok(SKFeatureBlock::from(Array2::zeros((2, 1))))
    }
}

pub(super) struct SKCscOutputStage;

impl SKPipelineTransformer<f64> for SKCscOutputStage {
    type Fitted = ();

    fn fit_stage(&self, _block: &SKFeatureBlock<f64>) -> Result<(), SKError> {
        Ok(())
    }

    fn transform_stage(
        &self,
        _fitted: &(),
        block: &SKFeatureBlock<f64>,
    ) -> Result<SKFeatureBlock<f64>, SKError> {
        let csc = CsMat::new_csc((block.number_of_rows(), 2), vec![0, 0, 0], vec![], vec![]);
        Ok(SKFeatureBlock::Sparse(csc))
    }
}
