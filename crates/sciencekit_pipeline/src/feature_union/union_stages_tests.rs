//! Tests for the `SKFeatureUnionStages` tuple trait and builder stage composition
//! (spec `feature-union`, Task 1.1).

use ndarray::Array2;
use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode, SKFeatureBlock};
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::feature_union::{SKFeatureUnion, SKFeatureUnionArity, SKFeatureUnionStages};

#[test]
fn two_member_scalar_chain_counts_members_and_fits_both_stages() {
    let standard = SKStandardScaler::build().build().unwrap();
    let minmax = SKMinMaxScaler::<f64>::build().build().unwrap();
    let union = SKFeatureUnion::build()
        .stages((standard, minmax))
        .build()
        .unwrap();

    assert_eq!(union.number_of_members(), 2);
    assert_eq!(union.backend(), SKBackendKind::Faer);
    assert_eq!(union.execution_mode(), SKExecutionMode::Automatic);

    let block = SKFeatureBlock::from(
        Array2::from_shape_vec((2, 2), vec![1.0_f64, 10.0, 3.0, 20.0]).unwrap(),
    );
    let (fitted_stages, widths) = union.stages().fit_stages(&block).unwrap();
    assert_eq!(widths, vec![2, 2]);
    assert!((fitted_stages.0.mean()[0] - 2.0).abs() < 1e-12);
    assert!((fitted_stages.1.minimum()[1] - 10.0).abs() < 1e-12);
}

#[test]
fn incremental_with_stage_builds_tuples_up_to_eight_members() {
    let make_scaler = || SKStandardScaler::build().build().unwrap();
    let union = SKFeatureUnion::build()
        .with_stage(make_scaler())
        .with_stage(make_scaler())
        .with_stage(make_scaler())
        .with_stage(make_scaler())
        .with_stage(make_scaler())
        .with_stage(make_scaler())
        .with_stage(make_scaler())
        .with_stage(make_scaler())
        .backend(SKBackendKind::Faer)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .unwrap();

    assert_eq!(union.stages().number_of_members(), 8);

    // Scalar-generic members work in both f32 and f64.
    let block_f32 =
        SKFeatureBlock::from(Array2::from_shape_vec((2, 1), vec![1.0_f32, 3.0]).unwrap());
    let (_, widths_f32) = union.stages().fit_stages(&block_f32).unwrap();
    assert_eq!(widths_f32, vec![1; 8]);
}
