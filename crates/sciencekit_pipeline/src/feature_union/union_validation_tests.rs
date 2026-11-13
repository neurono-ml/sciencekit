//! Validation, shape mismatch and member-index error tests for `SKFeatureUnion`
//! (spec `feature-union`, Task 2.2).

use ndarray::array;
use sciencekit_common::{SKError, SKFeatureBlock};
use sprs::CsMat;

use crate::SKFeatureUnion;

use super::union_validation_fixtures_tests::{
    SKCscOutputStage, SKFailingStage, SKFixedTwoRowStage, scaler,
};

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
            scaler(),
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
        .with_boxed_stage(scaler())
        .with_boxed_stage(scaler())
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

    let fitted = SKFeatureUnion::build()
        .stages((
            scaler(),
            SKFailingStage {
                fail_on_fit: false,
                fail_on_transform_rows: Some(1),
            },
        ))
        .build()
        .unwrap()
        .fit(&dense)
        .unwrap();

    let message = fitted
        .transform(&query_single_row)
        .err()
        .unwrap()
        .to_string();
    assert!(message.contains("feature union member 1"));

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
        .stages((scaler(), SKFixedTwoRowStage))
        .build()
        .unwrap();

    let fit_mismatch = union.fit_checked(&three_rows).err().unwrap();
    assert_eq!(fit_mismatch.member_index(), Some(1));
    assert!(matches!(
        fit_mismatch.error(),
        SKError::ShapeMismatch { .. }
    ));

    let fitted = union.fit(&two_rows).unwrap();
    let transform_mismatch = fitted.transform_checked(&three_rows).err().unwrap();
    assert_eq!(transform_mismatch.member_index(), Some(1));
    assert!(matches!(
        transform_mismatch.error(),
        SKError::ShapeMismatch { .. }
    ));

    let width_mismatch = fitted.transform_checked(&wrong_width).err().unwrap();
    assert_eq!(width_mismatch.member_index(), None);
    assert!(matches!(
        width_mismatch.error(),
        SKError::ShapeMismatch { .. }
    ));
}

#[test]
fn csc_member_output_is_rejected_with_unsupported_representation() {
    let training = SKFeatureBlock::from(array![[1.0_f64, 2.0], [3.0, 4.0]]);
    let csc_union = SKFeatureUnion::build()
        .stages((scaler(), SKCscOutputStage))
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
