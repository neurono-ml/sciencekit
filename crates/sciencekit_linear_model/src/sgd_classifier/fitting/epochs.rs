//! The in-memory multi-epoch one-vs-rest fit (spec decision 7).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKError, SKFloat};

use super::super::targets::{sk_head_targets, sk_positive_classes};
use super::driver::{SKFeatureDesign, sk_run_head_epoch, sk_score_rows};
use super::state::sk_reported_parts;
use crate::class_weight::{SKClassWeight, sk_expand_class_weights};
use crate::sgd::config::SKSGDConfig;
use crate::sgd::{
    SKConvergenceTracker, SKEarlyStopping, SKLearningState, SKRandomGenerator, SKSGDClassifierLoss,
    sk_finish_epochs, sk_objective_improved, sk_split_validation_indices,
};

/// The outcome of an in-memory fit: the reported iterate, the live states, the
/// convergence flag and the epochs executed.
pub(crate) struct SKEpochOutcome<F> {
    pub(crate) coefficients: Array2<F>,
    pub(crate) intercepts: Array1<F>,
    pub(crate) states: Vec<SKLearningState<F>>,
    pub(crate) converged: bool,
    pub(crate) epochs_taken: usize,
}

/// Run the epoch loop over a dense or CSR design.
pub(crate) fn sk_fit_epochs<F: SKFloat>(
    config: &SKSGDConfig<F>,
    loss: SKSGDClassifierLoss,
    class_weight: &SKClassWeight<F>,
    mut states: Vec<SKLearningState<F>>,
    class_indices: &[usize],
    number_of_classes: usize,
    design: SKFeatureDesign<'_, F>,
) -> Result<SKEpochOutcome<F>, SKError> {
    let sample_weights = sk_expand_class_weights(class_weight, class_indices, number_of_classes)?;
    let positive_classes = sk_positive_classes(number_of_classes);
    let targets: Vec<Vec<F>> = positive_classes
        .iter()
        .map(|&positive| sk_head_targets(class_indices, positive))
        .collect();
    let (training_rows, validation_rows) = sk_validation_split(config, design.rows())?;
    let mut generator = SKRandomGenerator::new(config.random_seed);
    let mut tracker = SKConvergenceTracker::new(
        config.tolerance,
        config.number_of_epochs_without_improvement,
    );
    let mut early = sk_early_stopping(config, states.len());
    let mut best: Vec<Option<(Array1<F>, F)>> = vec![None; states.len()];
    let mut previous_objective: Option<F> = None;
    let mut epochs_taken = 0;
    let mut converged = false;

    for _ in 0..config.maximum_number_of_iterations {
        epochs_taken += 1;
        let order = sk_epoch_order(config, &training_rows, &mut generator);
        let mut early_stop = false;
        let mut head_objectives = Vec::with_capacity(states.len());
        for head in 0..states.len() {
            let objective = sk_run_head_epoch(
                &mut states[head],
                design,
                ndarray::ArrayView1::from(targets[head].as_slice()),
                &sample_weights,
                loss,
                config,
                &order,
            )?;
            head_objectives.push(objective);
            if let Some(trackers) = &mut early {
                let scored = sk_score_rows(
                    &states[head],
                    design,
                    ndarray::ArrayView1::from(targets[head].as_slice()),
                    &sample_weights,
                    loss,
                    &validation_rows,
                );
                if trackers[head].record(
                    states[head].reported_weights(),
                    states[head].reported_intercept(),
                    scored,
                ) {
                    early_stop = true;
                }
                if let Some(weights) = trackers[head].best_weights() {
                    best[head] = Some((
                        weights.to_owned(),
                        trackers[head]
                            .best_intercept()
                            .expect("a best weight has its intercept"),
                    ));
                }
            }
        }
        let objective = head_objectives
            .iter()
            .fold(F::zero(), |accumulator, &value| accumulator + value)
            / F::from(states.len()).expect("the head count fits in the scalar");
        let improved = previous_objective
            .is_none_or(|previous| sk_objective_improved(previous, objective, config.tolerance));
        previous_objective = Some(objective);
        if !improved {
            states
                .iter_mut()
                .for_each(SKLearningState::halve_learning_rate);
        }
        converged = tracker.record(objective);
        if converged || early_stop {
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
    let (mut coefficients, mut intercepts) = sk_reported_parts(&states, design.columns());
    for (head, entry) in best.iter().enumerate() {
        if let Some((weights, intercept)) = entry {
            coefficients.row_mut(head).assign(weights);
            intercepts[head] = *intercept;
        }
    }
    Ok(SKEpochOutcome {
        coefficients,
        intercepts,
        states,
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

/// One early-stopping tracker per head, when the feature is enabled.
fn sk_early_stopping<F: SKFloat>(
    config: &SKSGDConfig<F>,
    number_of_heads: usize,
) -> Option<Vec<SKEarlyStopping<F>>> {
    config.early_stopping.then(|| {
        (0..number_of_heads)
            .map(|_| {
                SKEarlyStopping::new(
                    config.tolerance,
                    config.number_of_epochs_without_improvement,
                )
            })
            .collect()
    })
}
