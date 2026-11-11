//! Scripted classifier and span-recording fixtures for classifier bagging
//! tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKSupervisedFit, SKTargetView,
};
use tracing::field::{Field, Visit};
use tracing_subscriber::Layer;

#[derive(Clone, Debug)]
pub(super) struct ScriptedBaseClassifier {
    pub(super) call_counter: Arc<AtomicUsize>,
    pub(super) class_zero_probabilities: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(super) struct ScriptedClassifierModel {
    pub(super) class_zero_probability: f64,
}

impl SKSupervisedFit<f64> for ScriptedBaseClassifier {
    type Model = ScriptedClassifierModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, _features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        let slot = self.call_counter.fetch_add(1, Ordering::SeqCst);
        Ok(ScriptedClassifierModel {
            class_zero_probability: self.class_zero_probabilities[slot],
        })
    }
}

impl SKClassifierPredictor<f64> for ScriptedClassifierModel {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let probabilities = self.predict_probabilities(features)?;
        Ok(Array1::from_shape_fn(probabilities.nrows(), |row| {
            usize::from(probabilities[[row, 1]] > probabilities[[row, 0]]) as i64
        }))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array2::from_shape_fn((rows, 2), |(_, column)| {
            if column == 0 {
                self.class_zero_probability
            } else {
                1.0 - self.class_zero_probability
            }
        }))
    }
}

pub(super) struct BackendRecorder {
    pub(super) recorded: Arc<Mutex<Vec<String>>>,
}

struct BackendVisitor {
    backend: Option<String>,
}

impl Visit for BackendVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "backend" {
            self.backend = Some(format!("{value:?}"));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "backend" {
            self.backend = Some(value.to_string());
        }
    }
}

impl<S: tracing::Subscriber> Layer<S> for BackendRecorder {
    fn on_new_span(
        &self,
        attributes: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = BackendVisitor { backend: None };
        attributes.record(&mut visitor);
        if let Some(backend) = visitor.backend {
            self.recorded.lock().unwrap().push(backend);
        }
    }
}
