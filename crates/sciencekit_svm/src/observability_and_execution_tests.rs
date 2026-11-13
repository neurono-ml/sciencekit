//! Companion TDD tests for observability span recording, execution-mode plan
//! resolution, convergence iteration reporting, and `f32` support
//! (spec `support-vector-machines`, spec `backend-selection`).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKError, SKExecutionMode,
    SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_math::{SKKernel, SKKernelGamma};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

use crate::{SKSVC, SKSVR};

#[derive(Default)]
struct DebugFieldVisitor(HashMap<String, String>);

impl Visit for DebugFieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
}

#[derive(Clone, Default)]
struct SpanFieldRecorder {
    spans: Arc<Mutex<Vec<HashMap<String, String>>>>,
}

impl<S: Subscriber> Layer<S> for SpanFieldRecorder {
    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        let mut visitor = DebugFieldVisitor::default();
        attributes.values().record(&mut visitor);
        self.spans.lock().expect("span lock").push(visitor.0);
    }
}

#[test]
fn configured_backend_is_recorded_in_fit_and_prediction_spans() {
    let features = array![
        [-2.0_f64, -1.0],
        [-1.5, -0.5],
        [-1.0, -1.5],
        [1.0, 1.5],
        [1.5, 0.5],
        [2.0, 1.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    let targets = array![-2.0_f64, -1.5, -1.0, 1.0, 1.5, 2.0];

    let recorder = SpanFieldRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());

    tracing::subscriber::with_default(subscriber, || {
        let calibrated = SKSVC::<f64>::build()
            .kernel(SKKernel::Linear)
            .probability(true)
            .backend(SKBackendKind::MatrixMultiply)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap();
        let _ = calibrated.predict_labels(features.view()).unwrap();
        let _ = calibrated.predict_probabilities(features.view()).unwrap();
        let _ = calibrated.decision_function(features.view()).unwrap();
        let _ = calibrated
            .pairwise_decision_function(features.view())
            .unwrap();

        let uncalibrated = SKSVC::<f64>::build()
            .kernel(SKKernel::Linear)
            .backend(SKBackendKind::MatrixMultiply)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap();
        assert!(matches!(
            uncalibrated
                .predict_probabilities(features.view())
                .unwrap_err(),
            SKError::UnsupportedOperation { .. }
        ));

        let svr = SKSVR::<f64>::build()
            .kernel(SKKernel::Linear)
            .backend(SKBackendKind::MatrixMultiply)
            .build()
            .unwrap()
            .fit(features.view(), targets.view())
            .unwrap();
        assert_eq!(svr.backend(), SKBackendKind::MatrixMultiply);
        let _ = svr.predict(features.view()).unwrap();
    });

    let spans = recorder.spans.lock().expect("span lock");
    assert_eq!(spans.len(), 9, "recorded spans: {spans:?}");
    for span in spans.iter() {
        assert_eq!(
            span.get("backend").map(String::as_str),
            Some("matrixmultiply")
        );
    }
}

#[test]
fn out_of_core_streaming_execution_mode_is_rejected_on_fit() {
    let features = array![[-1.0_f64, 0.0], [1.0, 0.0]];
    let labels = array![0_i64, 1];
    let targets = array![-1.0_f64, 1.0];

    let svc_error = SKSVC::<f64>::build()
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(matches!(
        svc_error,
        SKError::ExecutionModeIncompatible { .. }
    ));

    let svr_error = SKSVR::<f64>::build()
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(
        svr_error,
        SKError::ExecutionModeIncompatible { .. }
    ));
}

#[test]
fn f32_svc_and_svr_fit_predict_report_iterations_and_svr_reports_not_converged() {
    let features = array![
        [-2.0_f32, -1.0],
        [-1.5, -0.5],
        [-1.0, -1.5],
        [1.0, 1.5],
        [1.5, 0.5],
        [2.0, 1.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    let targets = array![-2.0_f32, -1.4, -1.1, 1.1, 1.4, 2.0];

    let svc = SKSVC::<f32>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Scale)
        .probability(true)
        .random_seed(5)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert!(svc.converged());
    assert!(svc.iterations() > 0);
    assert_eq!(svc.predict_labels(features.view()).unwrap(), labels);
    let probabilities = svc.predict_probabilities(features.view()).unwrap();
    probabilities.rows().into_iter().for_each(|row| {
        let sum: f32 = row.iter().copied().sum();
        assert!((sum - 1.0_f32).abs() < 1e-4_f32);
    });

    let svr = SKSVR::<f32>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .epsilon(0.05)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(svr.converged());
    assert!(svr.iterations() > 0);
    let predicted = svr.predict(features.view()).unwrap();
    assert_eq!(predicted.len(), 6);

    let not_converged = SKSVR::<f32>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(50.0)
        .epsilon(1e-4)
        .tolerance(1e-6)
        .maximum_number_of_iterations(1)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(
        not_converged,
        SKError::NotConverged { iterations: 1 }
    ));
}
