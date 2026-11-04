//! The single-pass out-of-core regression fit (spec
//! `stochastic-gradient-regressor`, design decision 8).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKOperationAttributes, SKOperationKind,
    SKSupervisedLazySource, sk_run_operation,
};

use super::super::coefficients::SKSGDRegressorCoefficients;
use super::super::model::SKSGDRegressorModel;
use super::super::targets::sk_loss_adapter;
use super::state::{sk_create_state, sk_reported_parts};
use crate::sgd::config::SKSGDConfig;
use crate::sgd::{
    SKLearningState, SKRandomGenerator, SKSGDRegressorLoss, sk_finish_epochs, sk_run_batch_epoch,
};

/// Fit on a one-shot supervised source in a single pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_fit_streaming<F, S>(
    source: &mut S,
    config: &SKSGDConfig<F>,
    loss: SKSGDRegressorLoss,
    epsilon: F,
    warm_start: Option<&SKSGDRegressorCoefficients<F>>,
    backend: SKBackendKind,
    execution_intent: SKExecutionMode,
) -> Result<SKSGDRegressorModel<F>, SKError>
where
    F: SKFloat,
    S: SKSupervisedLazySource<F> + Send,
    S::Error: From<SKError> + Into<SKError>,
{
    if config.early_stopping {
        return Err(SKError::UnsupportedOperation {
            operation: "early_stopping",
            reason: "a one-shot stream has no epochs to compare; use an in-memory fit".to_string(),
        });
    }
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count: 0,
        columns_count: 0,
        execution_mode: execution_intent,
        backend,
    };
    sk_run_operation(attributes, || {
        let adapter = sk_loss_adapter(loss, epsilon);
        let mut generator = SKRandomGenerator::new(config.random_seed);
        let mut state: Option<SKLearningState<F>> = None;
        let mut columns = 0usize;
        let mut batches_seen = 0usize;
        while let Some(batch) = source.next_supervised_batch().map_err(Into::into)? {
            let features = batch.features();
            if features.nrows() == 0 {
                continue;
            }
            if columns != 0 && columns != features.ncols() {
                return Err(SKError::ShapeMismatch {
                    expected: vec![columns],
                    found: vec![features.ncols()],
                });
            }
            columns = features.ncols();
            batches_seen += 1;
            let targets = batch.targets();
            if targets.len() != features.nrows() {
                return Err(SKError::ShapeMismatch {
                    expected: vec![features.nrows()],
                    found: vec![targets.len()],
                });
            }
            if state.is_none() {
                state = Some(sk_create_state(
                    config,
                    loss,
                    epsilon,
                    warm_start,
                    columns,
                    targets.as_slice().expect("a batch target is contiguous"),
                )?);
            }
            let live = state
                .as_mut()
                .expect("the state exists after the first batch");
            let weights = vec![F::one(); features.nrows()];
            let order = sk_batch_order(config, features.nrows(), &mut generator);
            sk_run_batch_epoch(live, &batch, &weights, &adapter, config, &order)?;
        }
        if batches_seen == 0 {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        }
        sk_finish_epochs(
            config.convergence_policy,
            config.maximum_number_of_iterations,
            1,
            false,
            true,
        )?;
        let state = state.expect("a non-empty stream produced its state");
        let (coefficients, intercept) = sk_reported_parts(&state, columns);
        Ok(SKSGDRegressorModel::from_parts(
            coefficients,
            intercept,
            state,
            config,
            loss,
            epsilon,
            backend,
            execution_intent,
            false,
            1,
        ))
    })
}

/// Shuffle within a batch only: a one-shot source cannot be rewound.
fn sk_batch_order<F: SKFloat>(
    config: &SKSGDConfig<F>,
    rows: usize,
    generator: &mut SKRandomGenerator,
) -> Vec<usize> {
    if config.shuffle {
        generator.shuffled_indices(rows)
    } else {
        (0..rows).collect()
    }
}
