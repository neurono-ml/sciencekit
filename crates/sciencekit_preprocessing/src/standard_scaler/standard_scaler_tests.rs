//! Companion tests for the standard scaler: executable truth quoted by the
//! Development Explained book. Every test drives the public API only
//! (`build` → `fit` → `transform`), never private fields.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKDataBatch, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat,
    SKLazySource, SKUnsupervisedFit,
};

use crate::standard_scaler::core_implementation::SKStandardScaler;

fn default_scaler() -> SKStandardScaler {
    SKStandardScaler::build()
        .build()
        .expect("default scaler builds")
}

fn assert_close(actual: &Array1<f64>, expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (value, target) in actual.iter().zip(expected.iter()) {
        let tolerance = 1e-12 * (1.0 + value.abs().max(target.abs()));
        assert!(
            (value - target).abs() <= tolerance,
            "values {value} and {target} differ beyond tolerance {tolerance}"
        );
    }
}

/// Fit learns per-column mean and population scale.
#[test]
fn fit_learns_per_column_mean_and_scale() {
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.mean(), &[2.0, 3.0]);
    assert_close(model.scale(), &[1.0, 1.0]);
}

/// Transform centers and scales to z-scores.
#[test]
fn transform_produces_z_scores() {
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-1.0, -1.0], [1.0, 1.0]]);
}

/// `with_mean = false` learns zero means and leaves values uncentered.
#[test]
fn without_mean_keeps_values_uncentered() {
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let scaler = SKStandardScaler::build()
        .with_mean(false)
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_close(model.mean(), &[0.0, 0.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, features);
}

/// `with_std = false` learns unit scales and only centers.
#[test]
fn without_std_centers_without_scaling() {
    let features: Array2<f64> = array![[0.0, 0.0], [2.0, 4.0]];
    let scaler = SKStandardScaler::build()
        .with_std(false)
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_close(model.mean(), &[1.0, 2.0]);
    assert_close(model.scale(), &[1.0, 1.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-1.0, -2.0], [1.0, 2.0]]);
}

/// A constant column has zero variance: its scale is one and its transformed
/// values are zero instead of NaN.
#[test]
fn constant_column_yields_unit_scale_and_zero_output() {
    let features: Array2<f64> = array![[5.0, 1.0], [5.0, 3.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.scale(), &[1.0, 1.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[0.0, -1.0], [0.0, 1.0]]);
}

/// The estimator stays generic over both supported floats.
#[test]
fn fit_accepts_either_float() {
    fn mean_of<F: SKFloat>(raw: Vec<F>) -> Array1<F> {
        let features = Array2::from_shape_vec((2, 1), raw).unwrap();
        default_scaler()
            .fit(&features)
            .expect("fit succeeds")
            .mean()
            .clone()
    }
    assert_eq!(mean_of(vec![1.0_f32, 3.0]), array![2.0_f32]);
    assert_eq!(mean_of(vec![1.0_f64, 3.0]), array![2.0_f64]);
}

/// Empty inputs fail with a shape error before any element is processed.
#[test]
fn fit_rejects_empty_input() {
    let features: Array2<f64> = Array2::zeros((0, 2));
    let error = default_scaler()
        .fit(&features)
        .expect_err("empty fit must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Sparse inputs are rejected at the dense-only boundary with a precise error.
#[test]
fn fit_rejects_sparse_input() {
    let sparse = sprs::CsMat::new((2, 2), vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]);
    let error = default_scaler()
        .fit(sparse.view())
        .expect_err("sparse fit must fail");
    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
}

/// Transforming data with a different column count than the fitted model fails.
#[test]
fn transform_rejects_column_count_mismatch() {
    let fitted: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let other: Array2<f64> = array![[1.0, 2.0, 3.0]];
    let model = default_scaler().fit(&fitted).expect("fit succeeds");
    let error = model
        .transform(&other)
        .expect_err("mismatched transform must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The default backend is Faer on both estimator and model.
#[test]
fn backend_defaults_to_faer() {
    let scaler = default_scaler();
    assert_eq!(scaler.backend(), SKBackendKind::Faer);
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::Faer);
}

/// A backend override flows from builder into estimator and fitted model.
#[test]
fn backend_override_reaches_model() {
    let scaler = SKStandardScaler::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .expect("scaler builds");
    assert_eq!(scaler.backend(), SKBackendKind::MatrixMultiply);
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-1.0, -1.0], [1.0, 1.0]]);
}

/// An in-memory batch source: each vector entry is one owned batch.
struct VecSource<F> {
    batches: Vec<Array2<F>>,
    next: usize,
}

impl<F> VecSource<F> {
    fn new(batches: Vec<Array2<F>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKLazySource<F> for VecSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let data = self.batches[position].clone();
        let is_final = self.next == self.batches.len();
        Ok(Some(SKDataBatch::new(data, position, is_final)))
    }
}

fn six_by_two() -> Array2<f64> {
    array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]
}

/// Streaming over uneven batches learns the same model as the dense fit.
#[test]
fn streaming_matches_dense_fit() {
    let features = six_by_two();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let mut source = VecSource::new(vec![
        features.slice(ndarray::s![0..2, ..]).to_owned(),
        features.slice(ndarray::s![2..3, ..]).to_owned(),
        features.slice(ndarray::s![3..6, ..]).to_owned(),
    ]);
    let streamed = default_scaler()
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_close(streamed.mean(), &reference.mean().to_vec());
    assert_close(streamed.scale(), &reference.scale().to_vec());
    let output = streamed.transform(&features).expect("transform succeeds");
    let expected = reference.transform(&features).expect("transform succeeds");
    assert_eq!(output, expected);
}

/// A single-batch stream degenerates to the dense result.
#[test]
fn streaming_single_batch_matches_dense_fit() {
    let features = six_by_two();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let mut source = VecSource::new(vec![features.clone()]);
    let streamed = default_scaler()
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_close(streamed.mean(), &reference.mean().to_vec());
}

/// An empty source fails instead of producing a vacuous model.
#[test]
fn streaming_empty_source_errors() {
    let mut source: VecSource<f64> = VecSource::new(vec![]);
    let error = default_scaler()
        .fit_streaming(&mut source)
        .expect_err("empty stream must fail");
    assert!(matches!(error, SKError::Conversion(_)));
}

/// Batches disagreeing on column count fail with a shape error, keeping the
/// effects of earlier batches out of any returned model.
#[test]
fn streaming_column_mismatch_errors() {
    let mut source = VecSource::new(vec![
        array![[1.0, 2.0], [3.0, 4.0]],
        array![[5.0, 6.0, 7.0]],
    ]);
    let error = default_scaler()
        .fit_streaming(&mut source)
        .expect_err("mismatched stream must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The builder flags propagate through the streaming path.
#[test]
fn streaming_respects_builder_flags() {
    let features = six_by_two();
    let scaler = SKStandardScaler::build()
        .with_mean(false)
        .with_std(false)
        .build()
        .expect("scaler builds");
    let reference = scaler.fit(&features).expect("fit succeeds");
    let mut source = VecSource::new(vec![
        features.slice(ndarray::s![0..4, ..]).to_owned(),
        features.slice(ndarray::s![4..6, ..]).to_owned(),
    ]);
    let streamed = scaler
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_close(streamed.mean(), &reference.mean().to_vec());
    assert_close(streamed.scale(), &reference.scale().to_vec());
}

/// An explicit in-memory intent collects the batches through the dense path.
#[test]
fn streaming_explicit_in_process_collects_batches() {
    let features = six_by_two();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let scaler = SKStandardScaler::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("scaler builds");
    let mut source = VecSource::new(vec![
        features.slice(ndarray::s![0..2, ..]).to_owned(),
        features.slice(ndarray::s![2..6, ..]).to_owned(),
    ]);
    let collected = scaler
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_close(collected.mean(), &reference.mean().to_vec());
    assert_close(collected.scale(), &reference.scale().to_vec());
}

/// Streaming stays generic over both supported floats.
#[test]
fn streaming_accepts_either_float() {
    let batch_f32 = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let mut source = VecSource::new(vec![
        batch_f32.slice(ndarray::s![0..1, ..]).to_owned(),
        batch_f32.slice(ndarray::s![1..3, ..]).to_owned(),
    ]);
    let streamed = default_scaler()
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(streamed.mean().clone(), array![3.0_f32, 4.0]);
}
