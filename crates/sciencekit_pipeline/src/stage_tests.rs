//! Tests for the transformer-to-stage adapter (spec `typed-pipeline`).

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKFeatureBlock};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::SKPipelineTransformer;

/// A standard scaler over two columns fitted on `[[1, 2], [3, 4]]` learns mean
/// `[2, 3]` and unit scale, so `[[1, 2]]` maps to `[[-1, -1]]`.
fn fitted_scaler_stage() -> (SKStandardScaler, SKFeatureBlock<f64>) {
    let scaler = SKStandardScaler::build().build().unwrap();
    let block = SKFeatureBlock::from(
        Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    );
    (scaler, block)
}

fn dense_values(block: &SKFeatureBlock<f64>) -> Array2<f64> {
    match block.view() {
        SKDataView::Dense(view) => view.to_owned(),
        SKDataView::Sparse(_) => panic!("the standard scaler produces a dense block"),
        _ => panic!("unknown representation"),
    }
}

#[test]
fn scaler_stage_fits_and_transforms_a_dense_block() {
    let (scaler, training) = fitted_scaler_stage();
    let fitted = scaler.fit_stage(&training).unwrap();
    let transformed = scaler.transform_stage(&fitted, &training).unwrap();
    let values = dense_values(&transformed);
    assert!((values[[0, 0]] + 1.0).abs() < 1e-12);
    assert!((values[[0, 1]] + 1.0).abs() < 1e-12);
    assert!((values[[1, 0]] - 1.0).abs() < 1e-12);
    assert!((values[[1, 1]] - 1.0).abs() < 1e-12);
}

#[test]
fn scaler_stage_is_generic_over_the_scalar() {
    let scaler = SKStandardScaler::build().build().unwrap();
    let block = SKFeatureBlock::from(
        Array2::from_shape_vec((2, 1), vec![1.0_f32, 3.0]).unwrap(),
    );
    let fitted = scaler.fit_stage(&block).unwrap();
    let transformed = scaler.transform_stage(&fitted, &block).unwrap();
    assert_eq!(transformed.number_of_columns(), 1);
    assert!(!transformed.is_sparse());
}

#[test]
fn a_shape_mismatch_at_transform_is_surfaced() {
    let (scaler, training) = fitted_scaler_stage();
    let fitted = scaler.fit_stage(&training).unwrap();
    let wrong = SKFeatureBlock::from(
        Array2::from_shape_vec((1, 3), vec![1.0_f64, 2.0, 3.0]).unwrap(),
    );
    let error = scaler.transform_stage(&fitted, &wrong).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn a_sparse_input_rejection_keeps_its_structured_error() {
    let (scaler, training) = fitted_scaler_stage();
    let fitted = scaler.fit_stage(&training).unwrap();
    let sparse = SKFeatureBlock::Sparse(sprs::CsMat::new(
        (1, 2),
        vec![0, 1],
        vec![0],
        vec![1.0_f64],
    ));
    let error = scaler.transform_stage(&fitted, &sparse).unwrap_err();
    assert!(matches!(
        error,
        SKError::UnsupportedRepresentation { .. }
    ));
}
