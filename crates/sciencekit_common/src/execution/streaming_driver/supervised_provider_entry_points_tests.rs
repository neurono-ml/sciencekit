//! Tests for the supervised provider entry points (spec
//! `streaming-source-provider` and `streaming-batches`).

use super::provider_entry_point_test_sources::{RecordingSource, streaming_plan};
use super::{SKStreamDecision, SKStreamingPassPolicy, sk_run_supervised_streaming_provider};
use crate::batching::{SKReopenableSource, SKSupervisedBatch};

fn supervised_reopenable_provider_runs_the_budget() {
    let mut provider = SKReopenableSource::new(|| Ok(RecordingSource::<f64>::new(3)));
    let mut state = Vec::new();
    let update =
        |batch: SKSupervisedBatch<f64>, _parallelism: usize, state: &mut Vec<(usize, f64)>| {
            state.push((batch.position(), batch.targets()[0]));
            SKStreamDecision::Continue
        };

    let passes = sk_run_supervised_streaming_provider(
        &mut provider,
        &mut state,
        update,
        &streaming_plan(),
        SKStreamingPassPolicy::Iterative {
            maximum_number_of_passes: 2,
        },
    )
    .unwrap();

    assert_eq!(passes, 2);
    assert_eq!(
        state,
        vec![
            (0, 0.0),
            (1, 10.0),
            (2, 20.0),
            (0, 0.0),
            (1, 10.0),
            (2, 20.0)
        ],
        "each pass preserves block order and pairs targets"
    );
}

#[test]
fn supervised_reopenable_provider_runs_the_budget_f64() {
    supervised_reopenable_provider_runs_the_budget();
}
