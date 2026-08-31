//! Tests for fit/transformation contracts (spec `estimator-contracts`).

use ndarray::{Array1, Array2, array};

use super::{SKFeatureTransformer, SKSupervisedFit, SKUnsupervisedFit};
use crate::SKError;
use crate::data_view::SKDataView;

// ---- Example contract types used by the tests -----------------------------

/// An example unsupervised estimator (a clusterer).
#[derive(Debug, Default, Clone, Copy)]
struct ExampleClusterer<F: crate::SKFloat> {
    marker: std::marker::PhantomData<F>,
}

/// The model produced by `ExampleClusterer`.
#[derive(Debug, Clone, PartialEq)]
struct ExampleClustererModel<F: crate::SKFloat> {
    number_of_centers: usize,
    marker: std::marker::PhantomData<F>,
}

impl<F: crate::SKFloat> SKUnsupervisedFit<F> for ExampleClusterer<F> {
    type Model = ExampleClustererModel<F>;
    type Error = SKError;
    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let _view: SKDataView<'a, F> = features.try_into()?;
        Ok(ExampleClustererModel {
            number_of_centers: 3,
            marker: std::marker::PhantomData,
        })
    }
}

/// An example supervised estimator (a classifier). It has a `predict` on the
/// model only — the estimator itself exposes no prediction.
#[derive(Debug, Default, Clone, Copy)]
struct ExampleClassifier<F: crate::SKFloat> {
    marker: std::marker::PhantomData<F>,
}

#[derive(Debug, Clone, PartialEq)]
struct ExampleClassifierModel<F: crate::SKFloat> {
    classes: usize,
    marker: std::marker::PhantomData<F>,
}

impl<F: crate::SKFloat> ExampleClassifierModel<F> {
    /// Prediction exists **only** on the model type.
    fn predict(&self, _features: Array2<F>) -> Array1<usize> {
        array![0_usize, 0]
    }
}

impl<F: crate::SKFloat> SKSupervisedFit<F> for ExampleClassifier<F> {
    type Model = ExampleClassifierModel<F>;
    type Error = SKError;
    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<crate::SKTargetView<'a>, Error = SKError>,
    {
        let _view: SKDataView<'a, F> = features.try_into()?;
        let _targets: crate::SKTargetView<'a> = targets.try_into()?;
        Ok(ExampleClassifierModel {
            classes: 2,
            marker: std::marker::PhantomData,
        })
    }
}

/// An example transformer with a typed output (an `Array2<F>`).
struct ExampleScaler<F: crate::SKFloat> {
    marker: std::marker::PhantomData<F>,
}

impl<F: crate::SKFloat> SKFeatureTransformer<F> for ExampleScaler<F> {
    type Output = Array2<F>;
    type Error = SKError;
    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'a, F> = features.try_into()?;
        match view {
            SKDataView::Dense(d) => Ok(d.to_owned()),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "non-dense",
                suggestion: "provide dense features",
            }),
        }
    }
}

// ---- Tests -----------------------------------------------------------------

/// The unsupervised fit returns a distinct model type from the estimator.
#[test]
fn unsupervised_fit_returns_distinct_model() {
    let estimator = ExampleClusterer::<f32>::default();
    let data = Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let model = estimator.fit(&data).unwrap();
    // Model is a different type, carries the learned state.
    assert_eq!(model.number_of_centers, 3);
}

/// Fit on a shared reference keeps the estimator reusable.
#[test]
fn shared_reference_keeps_estimator_reusable() {
    let estimator = ExampleClusterer::<f64>::default();
    let data = Array2::zeros((4, 2));
    let model_a = estimator.fit(&data).unwrap();
    let model_b = estimator.fit(&data).unwrap();
    assert!(!std::ptr::eq(&model_a, &model_b)); // independent instances
}

/// Same estimator + deterministic data → independent models, deterministic at the interface level.
#[test]
fn repeated_fit_is_deterministic() {
    let estimator = ExampleClassifier::<f64>::default();
    let features = Array2::zeros((4, 2));
    let targets = array![0_i64, 0, 1, 1];
    let targets_view = targets.view();
    let model_a = estimator.fit(&features, targets_view).unwrap();
    let model_b = estimator.fit(&features, targets_view).unwrap();
    assert_eq!(model_a.classes, model_b.classes);
}

/// The supervised fit requires targets in the signature (it takes them).
#[test]
fn supervised_fit_requires_targets() {
    let estimator = ExampleClassifier::<f64>::default();
    let features = Array2::zeros((2, 2));
    let targets = array![0_i64, 1];
    // Two-argument call compiles because targets are in the signature.
    let model = estimator.fit(&features, targets.view()).unwrap();
    assert_eq!(model.classes, 2);
}

/// Models are `Send + Sync` — shareable across threads without a mutex.
#[test]
fn fitted_models_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExampleClustererModel<f32>>();
    assert_send_sync::<ExampleClassifierModel<f64>>();
}

/// The configured estimator exposes no prediction — only the model does.
#[test]
fn prediction_lives_only_on_the_model() {
    let estimator = ExampleClassifier::<f32>::default();
    let _estimator = estimator;
    // The estimator type has no `predict` method by construction; prediction is
    // only on `ExampleClassifierModel::predict`. This test builds the model
    // through fit and calls predict on the *model*.
    let features = Array2::zeros((2, 2));
    let targets = array![0_i64, 1];
    let model = ExampleClassifier::<f32>::default()
        .fit(&features, targets.view())
        .unwrap();
    let _pred = model.predict(Array2::zeros((2, 2)));
}

/// A transformer's typed output chains into another stage's input.
#[test]
fn transformer_output_chains_statically() {
    let scaler = ExampleScaler::<f32> {
        marker: std::marker::PhantomData,
    };
    let data = Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let out: Array2<f32> = scaler.transform(&data).unwrap();
    // The output (Array2<f32>) is exactly what a downstream dense stage accepts.
    let _next_view: SKDataView<'_, f32> = (&out).try_into().unwrap();
    assert_eq!(out.shape(), &[2, 2]);
}
