//! Companion tests for `SKIsolationForest` fitting, subsampling, seeding and model state.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};
use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

use crate::{
    SK_ISOLATION_AUTO_OFFSET, SKContamination, SKIsolationForest, SKIsolationMaximumSamples,
};

#[derive(Clone, Default)]
struct SpanRecorder {
    spans: Arc<Mutex<Vec<HashMap<String, String>>>>,
}

struct FieldVisitor {
    fields: HashMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.fields
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

impl<S> Layer<S> for SpanRecorder
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        let mut visitor = FieldVisitor {
            fields: HashMap::new(),
        };
        attributes.values().record(&mut visitor);
        self.spans.lock().unwrap().push(visitor.fields);
    }
}

fn dense_to_csr(matrix: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new((matrix.nrows(), matrix.ncols()));
    matrix
        .rows()
        .into_iter()
        .enumerate()
        .for_each(|(row_index, row_values)| {
            row_values
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value != 0.0)
                .for_each(|(column_index, &value)| {
                    triplets.add_triplet(row_index, column_index, value);
                });
        });
    triplets.to_csr()
}

#[test]
fn isolation_fit_builds_seeded_ensemble_and_transfers_model_state() {
    let data = array![
        [0.0_f64, 0.1, -0.2],
        [0.2, -0.1, 0.0],
        [-0.1, 0.2, 0.1],
        [0.1, 0.0, -0.1],
        [0.0, -0.2, 0.2],
        [12.0, -15.0, 9.0]
    ];
    let detector = SKIsolationForest::<f64>::build()
        .number_of_trees(24)
        .maximum_samples(SKIsolationMaximumSamples::Count(4))
        .contamination(SKContamination::Auto)
        .maximum_features(0.75)
        .bootstrap(false)
        .random_seed(17)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    let recorder = SpanRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    let model = tracing::subscriber::with_default(subscriber, || {
        tracing::callsite::rebuild_interest_cache();
        SKUnsupervisedFit::fit(&detector, &data).unwrap()
    });

    assert_eq!(model.number_of_trees(), 24);
    assert_eq!(model.feature_count(), 3);
    assert_eq!(model.subsample_size(), 4);
    assert!((model.offset() - SK_ISOLATION_AUTO_OFFSET).abs() < 1e-15);
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
    assert!(model.tree(24).is_none());

    (0..24).for_each(|tree_index| {
        let tree = model.tree(tree_index).unwrap();
        assert_eq!(tree.feature_count(), 3);
        assert!(tree.depth() <= 2);
        assert_eq!(tree.node(tree.root_index()).unwrap().sample_count(), 4);
    });
    assert_ne!(model.tree(0), model.tree(1));

    let spans = recorder.spans.lock().unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].get("operation").map(String::as_str), Some("fit"));
    assert_eq!(spans[0].get("rows").map(String::as_str), Some("6"));
    assert_eq!(spans[0].get("columns").map(String::as_str), Some("3"));
    assert_eq!(
        spans[0].get("backend").map(String::as_str),
        Some("matrixmultiply")
    );
}

#[test]
fn isolation_fit_sparse_and_dense_produce_identical_ensembles_and_offsets() {
    let data = array![
        [0.0_f64, 1.0, 0.0, -0.5],
        [0.1, 0.9, 0.0, -0.4],
        [0.0, 1.1, 0.2, -0.6],
        [-0.1, 1.0, 0.0, -0.5],
        [0.0, 0.8, -0.1, -0.5],
        [0.2, 1.0, 0.0, -0.3],
        [0.0, 1.2, 0.1, -0.5],
        [25.0, -30.0, 18.0, 0.0]
    ];
    let csr = dense_to_csr(&data);

    for bootstrap in [false, true] {
        let detector = SKIsolationForest::<f64>::build()
            .number_of_trees(16)
            .maximum_samples(SKIsolationMaximumSamples::Count(6))
            .contamination(SKContamination::Fraction(0.125))
            .maximum_features(0.75)
            .bootstrap(bootstrap)
            .random_seed(2026)
            .build()
            .unwrap();

        let dense_model = detector.fit(&data).unwrap();
        let sparse_model = detector.fit(csr.view()).unwrap();

        assert_eq!(
            dense_model.number_of_trees(),
            sparse_model.number_of_trees()
        );
        assert_eq!(dense_model.subsample_size(), sparse_model.subsample_size());
        assert!((dense_model.offset() - sparse_model.offset()).abs() < 1e-15);
        (0..16).for_each(|tree_index| {
            assert_eq!(dense_model.tree(tree_index), sparse_model.tree(tree_index));
        });
    }
}
