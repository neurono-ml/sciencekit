//! Tests for fit/transformation contracts (spec `estimator-contracts`).

use ndarray::{Array1, Array2, array};

use super::{SKClassifierPredictor, SKFeatureTransformer, SKRegressorPredictor};
use super::{SKSupervisedFit, SKUnsupervisedFit};
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

/// An example regressor model predicting a constant response in the model scalar.
#[derive(Debug, Clone, PartialEq)]
struct ExampleRegressorModel<F: crate::SKFloat> {
    response: F,
}

impl<F: crate::SKFloat> SKRegressorPredictor<F> for ExampleRegressorModel<F> {
    type Error = SKError;
    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'a, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        Ok(Array1::from_elem(rows, self.response))
    }
}

/// An example classifier model predicting constant labels and probabilities.
#[derive(Debug, Clone, PartialEq)]
struct ExampleClassifierModel<F: crate::SKFloat> {
    classes: usize,
    marker: std::marker::PhantomData<F>,
}

impl<F: crate::SKFloat> SKClassifierPredictor<F> for ExampleClassifierModel<F> {
    type Error = SKError;
    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'a, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        Ok(Array1::from_elem(rows, 1_i64))
    }
    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'a, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        let mut probabilities = Array2::zeros((rows, self.classes));
        probabilities.column_mut(1).fill(F::one());
        Ok(probabilities)
    }
}

impl<F: crate::SKFloat> SKSupervisedFit<F> for ExampleClassifier<F> {
    type Model = ExampleClassifierModel<F>;
    type Error = SKError;
    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<crate::SKTargetView<'a, F>, Error = SKError>,
    {
        let _view: SKDataView<'a, F> = features.try_into()?;
        let _targets: crate::SKTargetView<'a, F> = targets.try_into()?;
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
    assert_send_sync::<ExampleRegressorModel<f32>>();
}

/// Concurrent prediction shares one fitted model across threads per scalar.
#[test]
fn concurrent_predict_shares_one_model() {
    let model = ExampleRegressorModel { response: 1.0_f32 };
    let features = Array2::zeros((8, 2));
    let predictions: Vec<Array1<f32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    model
                        .predict(features.view())
                        .expect("thread-local predict succeeds")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("thread joins"))
            .collect()
    });
    assert_eq!(predictions.len(), 4);
    for prediction in predictions {
        assert_eq!(prediction.as_slice().unwrap(), &[1.0_f32; 8]);
    }
}

/// Predictions run on a single row and on a large batch in both scalars.
#[test]
fn predict_runs_on_small_and_large_batches() {
    let regressor = ExampleRegressorModel { response: 3.0_f64 };
    let single = Array2::zeros((1, 2));
    assert_eq!(regressor.predict(single.view()).unwrap().len(), 1);
    let large = Array2::zeros((10_000, 4));
    let predictions = regressor.predict(large.view()).unwrap();
    assert_eq!(predictions.len(), 10_000);

    let classifier = ExampleClassifierModel::<f32> {
        classes: 2,
        marker: std::marker::PhantomData,
    };
    let single = Array2::zeros((1, 2));
    assert_eq!(classifier.predict_labels(single.view()).unwrap().len(), 1);
    let large = Array2::zeros((10_000, 4));
    assert_eq!(
        classifier.predict_labels(large.view()).unwrap().len(),
        10_000
    );
}

/// The configured estimator exposes no prediction — only the model does.
#[test]
fn prediction_lives_only_on_the_model() {
    let estimator = ExampleClassifier::<f32>::default();
    let _estimator = estimator;
    // The estimator type has no `predict_labels` method by construction;
    // prediction is only on `ExampleClassifierModel`. This test builds the
    // model through fit and calls prediction on the *model*.
    let features = Array2::zeros((2, 2));
    let targets = array![0_i64, 1];
    let model = ExampleClassifier::<f32>::default()
        .fit(&features, targets.view())
        .unwrap();
    let labels = model.predict_labels(features.view()).unwrap();
    assert_eq!(labels.as_slice().unwrap(), &[1_i64, 1]);
}

/// Regressor predictions preserve the model scalar end to end.
#[test]
fn regressor_predictions_preserve_scalar() {
    let model = ExampleRegressorModel { response: 2.5_f32 };
    let features = Array2::zeros((3, 2));
    let predictions = model.predict(features.view()).unwrap();
    assert_eq!(predictions.as_slice().unwrap(), &[2.5_f32; 3]);
}

/// Classifier labels are dtype-independent; probabilities keep the scalar.
#[test]
fn classifier_labels_are_scalar_independent() {
    let model = ExampleClassifierModel::<f32> {
        classes: 2,
        marker: std::marker::PhantomData,
    };
    let features = Array2::zeros((2, 3));
    let labels = model.predict_labels(features.view()).unwrap();
    assert_eq!(labels.as_slice().unwrap(), &[1_i64, 1]);
    let probabilities = model.predict_probabilities(features.view()).unwrap();
    assert_eq!(probabilities.shape(), &[2, 2]);
    assert_eq!(probabilities.row(0).as_slice().unwrap(), &[0.0_f32, 1.0]);
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

/// A homogeneous scalar flows from transformer output into regressor input.
#[test]
fn homogeneous_scalar_flows_across_stages() {
    let scaler = ExampleScaler::<f32> {
        marker: std::marker::PhantomData,
    };
    let regressor = ExampleRegressorModel { response: 9.0_f32 };
    let data = Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let scaled: Array2<f32> = scaler.transform(&data).unwrap();
    let predictions = regressor.predict(scaled.view()).unwrap();
    assert_eq!(predictions.as_slice().unwrap(), &[9.0_f32, 9.0]);
}
