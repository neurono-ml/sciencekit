//! Validation, shape mismatch and member-index error tests for `SKFeatureUnion`
//! (spec `feature-union`, Task 2.2).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKFeatureBlock};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sprs::CsMat;

use crate::{SKFeatureUnion, SKPipelineTransformer};

struct SKFailingStage {
    fail_on_fit: bool,
    fail_on_transform_rows: Option<usize>,
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

struct SKFixedTwoRowStage;

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

struct SKCscOutputStage;

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

#[test]
fn empty_tuple_and_empty_boxed_unions_are_rejected_with_invalid_hyperparameter() {
    assert!(matches!(
        SKFeatureUnion::build().build(),
        Err(SKError::InvalidHyperparameter { name: "stages", .. })
    ));
    assert!(matches!(
        SKFeatureUnion::boxed::<f64>().build(),
        Err(SKError::InvalidHyperparameter { name: "stages", .. })
    ));
}

#[test]
fn member_failure_in_fit_wraps_member_index_in_tuple_and_boxed_unions() {
    let input = SKFeatureBlock::from(array![[1.0_f64, 2.0], [3.0, 4.0]]);
    let tuple_union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKFailingStage {
                fail_on_fit: true,
                fail_on_transform_rows: None,
            },
        ))
        .build()
        .unwrap();

    let error = tuple_union.fit(&input).err().unwrap();
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
    assert!(error.to_string().contains("feature union member 1"));

    let boxed_union = SKFeatureUnion::boxed::<f64>()
        .with_boxed_stage(SKStandardScaler::build().build().unwrap())
        .with_boxed_stage(SKStandardScaler::build().build().unwrap())
        .with_boxed_stage(SKFailingStage {
            fail_on_fit: true,
            fail_on_transform_rows: None,
        })
        .build()
        .unwrap();

    let boxed_error = boxed_union.fit_checked(&input).err().unwrap();
    assert_eq!(boxed_error.member_index(), Some(2));
    assert!(boxed_error.to_string().contains("feature union member 2"));
}

#[test]
fn member_failure_in_transform_surfaces_member_index_and_preserves_structured_errors() {
    let dense = SKFeatureBlock::from(array![[1.0_f64, 0.0], [0.0, 2.0]]);
    let query_single_row = SKFeatureBlock::from(array![[1.0_f64, 2.0]]);
    let sparse = SKFeatureBlock::Sparse(CsMat::new(
        (2, 2),
        vec![0, 1, 2],
        vec![0, 1],
        vec![1.0, 2.0],
    ));

    let union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKFailingStage {
                fail_on_fit: false,
                fail_on_transform_rows: Some(1),
            },
        ))
        .build()
        .unwrap();
    let fitted = union.fit(&dense).unwrap();

    let transform_err = fitted.transform(&query_single_row).err().unwrap();
    assert!(transform_err.to_string().contains("feature union member 1"));

    let checked = fitted.transform_checked(&sparse).err().unwrap();
    assert_eq!(checked.member_index(), Some(0));
    assert!(matches!(
        checked.error(),
        SKError::UnsupportedRepresentation { .. }
    ));
}

#[test]
fn row_count_mismatch_and_feature_count_mismatch_return_shape_mismatch() {
    let two_rows = SKFeatureBlock::from(array![[1.0_f64, 2.0], [3.0, 4.0]]);
    let three_rows = SKFeatureBlock::from(array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let wrong_width = SKFeatureBlock::from(array![[1.0_f64, 2.0, 3.0]]);

    let union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKFixedTwoRowStage,
        ))
        .build()
        .unwrap();

    assert!(matches!(
        union.fit(&three_rows),
        Err(SKError::ShapeMismatch { .. })
    ));
    let fitted = union.fit(&two_rows).unwrap();
    assert!(matches!(
        fitted.transform(&three_rows),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        fitted.transform(&wrong_width),
        Err(SKError::ShapeMismatch { .. })
    ));
}

#[test]
fn csc_member_output_is_rejected_with_unsupported_representation() {
    let training = SKFeatureBlock::from(array![[1.0_f64, 2.0], [3.0, 4.0]]);
    let csc_union = SKFeatureUnion::build()
        .stages((SKStandardScaler::build().build().unwrap(), SKCscOutputStage))
        .build()
        .unwrap();
    assert!(matches!(
        csc_union.fit(&training),
        Err(SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        })
    ));
}
