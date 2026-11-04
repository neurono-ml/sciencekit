//! The in-memory multi-epoch regression fit (spec decision 7).

use ndarray::Array1;
use sciencekit_common::{SKError, SKFloat};

use super::design::{SKRegressorDesign, sk_run_design_epoch, sk_score_rows};
use super::state::sk_reported_parts;
use crate::sgd::config::SKSGDConfig;
use crate::sgd::{
    SKConvergenceTracker, SKEarlyStopping, SKLearningState, SKRandomGenerator,
    SKSGDRegressorLossAdapter, sk_finish_epochs, sk_objective_improved,
    sk_split_validation_indices,
};

/// The outcome of an in-memory fit: the reported iterate, the live state, the
/// convergence flag and the epochs executed.
pub(crate) struct SKEpochOutcome<F> {
    pub(crate) coefficients: Array1<F>,
    pub(crate) intercept: F,
    pub(crate) state: SKLearningState<F>,
    pub(crate) converged: bool,
    pub(crate) epochs_taken: usize,
}

/// Run the epoch loop over a dense or CSR design.
pub(crate) fn sk_fit_epochs<F: SKFloat>(
    config: &SKSGDConfig<F>,
    loss: &SKSGDRegressorLossAdapter<F>,
    mut state: SKLearningState<F>,
    design: SKRegressorDesign<'_, F>,
    targets: &[F],
) -> Result<SKEpochOutcome<F>, SKError> {
    let sample_weights = vec![F::one(); design.rows()];
    let targets = Array1::from_vec(targets.to_vec());
    let (training_rows, validation_rows) = sk_validation_split(config, design.rows())?;
    let mut generator = SKRandomGenerator::new(config.random_seed);
    let mut tracker = SKConvergenceTracker::new(
        config.tolerance,
        config.number_of_epochs_without_improvement,
    );
    let mut early = config.early_stopping.then(|| {
        SKEarlyStopping::new(
            config.tolerance,
            config.number_of_epochs_without_improvement,
        )
    });
    let mut best: Option<(Array1<F>, F)> = None;
    let mut previous_objective: Option<F> = None;
    let mut epochs_taken = 0;
    let mut converged = false;
    let mut early_stopped = false;

    for _ in 0..config.maximum_number_of_iterations {
        epochs_taken += 1;
        let order = sk_epoch_order(config, &training_rows, &mut generator);
        let objective = sk_run_design_epoch(
            &mut state,
            design,
            targets.view(),
            &sample_weights,
            loss,
            config,
            &order,
        )?;
        let mut early_stop = false;
        if let Some(tracker) = &mut early {
            let scored = sk_score_rows(
                &state,
                design,
                targets.view(),
                &sample_weights,
                loss,
                &validation_rows,
            );
            early_stop =
                tracker.record(state.reported_weights(), state.reported_intercept(), scored);
            if let Some(weights) = tracker.best_weights() {
                best = Some((
                    weights.to_owned(),
                    tracker
                        .best_intercept()
                        .expect("a best weight has its intercept"),
                ));
            }
        }
        let improved = previous_objective
            .is_none_or(|previous| sk_objective_improved(previous, objective, config.tolerance));
        previous_objective = Some(objective);
        if !improved {
            state.halve_learning_rate();
        }
        converged = tracker.record(objective) || early_stop;
        early_stopped = early_stop;
        if converged {
            break;
        }
    }

    sk_finish_epochs(
        config.convergence_policy,
        config.maximum_number_of_iterations,
        epochs_taken,
        converged,
        false,
    )?;
    let (mut coefficients, mut intercept) = sk_reported_parts(&state, design.columns());
    if let Some((weights, best_intercept)) = &best {
        if early_stopped || converged {
            state.reset_iterate(weights.view(), *best_intercept);
        }
        coefficients = weights.clone();
        intercept = *best_intercept;
    }
    Ok(SKEpochOutcome {
        coefficients,
        intercept,
        state,
        converged,
        epochs_taken,
    })
}

/// The training/validation row split, empty validation without early stopping.
fn sk_validation_split<F: SKFloat>(
    config: &SKSGDConfig<F>,
    rows: usize,
) -> Result<(Vec<usize>, Vec<usize>), SKError> {
    if !config.early_stopping {
        return Ok(((0..rows).collect(), Vec::new()));
    }
    if rows < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "early_stopping",
            reason: "early stopping requires at least two rows".to_string(),
        });
    }
    Ok(sk_split_validation_indices(
        rows,
        config.validation_fraction,
    ))
}

/// The per-epoch visiting order: shuffled with the seeded generator, or the
/// input order when shuffling is disabled.
fn sk_epoch_order<F: SKFloat>(
    config: &SKSGDConfig<F>,
    training_rows: &[usize],
    generator: &mut SKRandomGenerator,
) -> Vec<usize> {
    if config.shuffle {
        generator
            .shuffled_indices(training_rows.len())
            .into_iter()
            .map(|position| training_rows[position])
            .collect()
    } else {
        training_rows.to_vec()
    }
}
