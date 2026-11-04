//! Tests for the feature-only provider entry points (spec
//! `streaming-source-provider` and `streaming-batches`).

use std::cell::Cell;
use std::rc::Rc;

use super::provider_entry_point_test_sources::{RecordingSource, streaming_plan};
use super::{SKStreamDecision, SKStreamingPassPolicy, sk_run_streaming_provider};
use crate::SKError;
use crate::batching::SKDataBatch;
use crate::batching::SKReopenableSource;

fn collect_update(
    batch: SKDataBatch<f64>,
    _parallelism: usize,
    state: &mut Vec<usize>,
) -> SKStreamDecision {
    state.push(batch.position());
    SKStreamDecision::Continue
}

fn one_shot_provider_runs_one_pass() {
    let mut source = RecordingSource::<f64>::new(3);
    let mut provider = &mut source;
    let mut state = Vec::new();

    let passes = sk_run_streaming_provider(
        &mut provider,
        &mut state,
        collect_update,
        &streaming_plan(),
        SKStreamingPassPolicy::Iterative {
            maximum_number_of_passes: 5,
        },
    )
    .unwrap();

    assert_eq!(passes, 1, "a one-shot provider must not be reopened");
    assert_eq!(state, vec![0, 1, 2]);
}

fn reopenable_provider_runs_the_budget() {
    let opens = Rc::new(Cell::new(0));
    let counter = Rc::clone(&opens);
    let mut provider = SKReopenableSource::new(move || {
        counter.set(counter.get() + 1);
        Ok(RecordingSource::<f64>::new(3))
    });
    let mut state = Vec::new();

    let passes = sk_run_streaming_provider(
        &mut provider,
        &mut state,
        collect_update,
        &streaming_plan(),
        SKStreamingPassPolicy::Iterative {
            maximum_number_of_passes: 3,
        },
    )
    .unwrap();

    assert_eq!(passes, 3);
    assert_eq!(opens.get(), 3);
    assert_eq!(state, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
}

fn single_pass_policy_runs_one_pass_on_reopenable() {
    let opens = Rc::new(Cell::new(0));
    let counter = Rc::clone(&opens);
    let mut provider = SKReopenableSource::new(move || {
        counter.set(counter.get() + 1);
        Ok(RecordingSource::<f64>::new(3))
    });
    let mut state = Vec::new();

    let passes = sk_run_streaming_provider(
        &mut provider,
        &mut state,
        collect_update,
        &streaming_plan(),
        SKStreamingPassPolicy::SinglePass,
    )
    .unwrap();

    assert_eq!(passes, 1);
    assert_eq!(opens.get(), 1);
    assert_eq!(state, vec![0, 1, 2], "an accumulator sees the data once");
}

fn a_stop_in_the_first_pass_skips_the_remaining_passes() {
    let opens = Rc::new(Cell::new(0));
    let counter = Rc::clone(&opens);
    let mut provider = SKReopenableSource::new(move || {
        counter.set(counter.get() + 1);
        Ok(RecordingSource::<f64>::new(3))
    });
    let mut state = Vec::new();
    let stop_after_first =
        |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<usize>| {
            state.push(batch.position());
            SKStreamDecision::Stop
        };

    let passes = sk_run_streaming_provider(
        &mut provider,
        &mut state,
        stop_after_first,
        &streaming_plan(),
        SKStreamingPassPolicy::Iterative {
            maximum_number_of_passes: 3,
        },
    )
    .unwrap();

    assert_eq!(passes, 1, "a stop must abort the remaining passes");
    assert_eq!(opens.get(), 1, "the factory must not be called again");
    assert_eq!(state, vec![0]);
}

fn factory_failure_aborts_the_fit() {
    let mut calls = 0usize;
    let mut provider = SKReopenableSource::new(move || {
        calls += 1;
        if calls == 2 {
            Err(SKError::Conversion(
                "factory failed on the second pass".into(),
            ))
        } else {
            Ok(RecordingSource::<f64>::new(2))
        }
    });
    let mut state = Vec::new();

    let result = sk_run_streaming_provider(
        &mut provider,
        &mut state,
        collect_update,
        &streaming_plan(),
        SKStreamingPassPolicy::Iterative {
            maximum_number_of_passes: 3,
        },
    );

    let error = match result {
        Ok(_) => panic!("a factory failure must abort the fit"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("factory failed on the second pass")
    );
    assert_eq!(state, vec![0, 1], "the first pass's effects are preserved");
}

#[test]
fn one_shot_provider_runs_one_pass_f64() {
    one_shot_provider_runs_one_pass();
}

#[test]
fn reopenable_provider_runs_the_budget_f64() {
    reopenable_provider_runs_the_budget();
}

#[test]
fn single_pass_policy_runs_one_pass_on_reopenable_f64() {
    single_pass_policy_runs_one_pass_on_reopenable();
}

#[test]
fn factory_failure_aborts_the_fit_f64() {
    factory_failure_aborts_the_fit();
}

#[test]
fn a_stop_in_the_first_pass_skips_the_remaining_passes_f64() {
    a_stop_in_the_first_pass_skips_the_remaining_passes();
}
