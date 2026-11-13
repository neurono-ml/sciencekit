//! Seeded validation splitting and patience-based early-stopping monitor for
//! gradient boosting.

use sciencekit_common::{
    SK_RANDOM_DOMAIN_FOLD, SK_RANDOM_DOMAIN_TREE, SKError, SKFloat, SKRandomGenerator,
    sk_derive_seed,
};

/// Configuration of the held-out validation split and patience monitor.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKEarlyStoppingConfig<F> {
    pub(crate) validation_fraction: F,
    pub(crate) patience: usize,
    pub(crate) tolerance: F,
    pub(crate) random_seed: u64,
}

/// Tracks validation loss across fitted stages and remembers the best stage
/// count so overfitting stages can be truncated on exit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKEarlyStoppingMonitor<F> {
    patience: usize,
    tolerance: F,
    best_validation_loss: Option<F>,
    best_stage_count: usize,
    stages_without_improvement: usize,
}

impl<F: SKFloat> SKEarlyStoppingMonitor<F> {
    /// Construct a fresh monitor with `patience` (`n_iter_no_change`) and
    /// minimum loss improvement `tolerance`.
    pub(crate) fn new(patience: usize, tolerance: F) -> Self {
        Self {
            patience,
            tolerance,
            best_validation_loss: None,
            best_stage_count: 1,
            stages_without_improvement: 0,
        }
    }

    /// Record the validation loss after `stage_count` stages; returns `true`
    /// when `patience` non-improving stages have elapsed.
    pub(crate) fn record(&mut self, stage_count: usize, validation_loss: F) -> bool {
        let improved = self
            .best_validation_loss
            .is_none_or(|best| validation_loss < best - self.tolerance);
        if improved {
            self.best_validation_loss = Some(validation_loss);
            self.best_stage_count = stage_count;
            self.stages_without_improvement = 0;
            return false;
        }
        self.stages_without_improvement += 1;
        self.stages_without_improvement >= self.patience
    }

    /// The 1-based stage count that achieved the lowest validation loss.
    pub(crate) fn best_stage_count(&self) -> usize {
        self.best_stage_count
    }
}

/// Partition `0..row_count` into `(training_indices, validation_indices)` using
/// a seeded draw without replacement.
pub(crate) fn sk_split_regression_validation_rows<F: SKFloat>(
    row_count: usize,
    validation_fraction: F,
    global_seed: u64,
) -> Result<(Vec<usize>, Vec<usize>), SKError> {
    if row_count < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "validation_fraction",
            reason: "early stopping requires at least two samples to hold out a validation split"
                .to_string(),
        });
    }
    let fraction = validation_fraction
        .to_f64()
        .expect("validation fraction fits in f64");
    let validation_count = ((fraction * row_count as f64).round() as usize).clamp(1, row_count - 1);
    let fold_seed = sk_derive_seed(global_seed, SK_RANDOM_DOMAIN_FOLD, 0);
    let mut validation_indices = SKRandomGenerator::new(fold_seed)
        .sample_indices_without_replacement(row_count, validation_count);
    validation_indices.sort_unstable();
    let training_indices = sk_complement_indices(row_count, &validation_indices);
    Ok((training_indices, validation_indices))
}

/// Partition `0..class_indices.len()` into stratified `(training_indices,
/// validation_indices)` so every canonical class retains at least one training
/// sample.
pub(crate) fn sk_split_classification_validation_rows<F: SKFloat>(
    class_indices: &[usize],
    number_of_classes: usize,
    validation_fraction: F,
    global_seed: u64,
) -> Result<(Vec<usize>, Vec<usize>), SKError> {
    let row_count = class_indices.len();
    if row_count <= number_of_classes {
        return Err(SKError::InvalidHyperparameter {
            name: "validation_fraction",
            reason:
                "early stopping requires more samples than classes to hold out a validation split"
                    .to_string(),
        });
    }
    let mut per_class_rows = vec![Vec::new(); number_of_classes];
    class_indices
        .iter()
        .enumerate()
        .for_each(|(row, &class_index)| per_class_rows[class_index].push(row));
    let fraction = validation_fraction
        .to_f64()
        .expect("validation fraction fits in f64");
    let mut per_class_validation: Vec<usize> = per_class_rows
        .iter()
        .map(|rows| {
            if rows.len() <= 1 {
                0
            } else {
                ((fraction * rows.len() as f64).round() as usize).clamp(0, rows.len() - 1)
            }
        })
        .collect();
    if per_class_validation.iter().sum::<usize>() == 0 {
        let largest_class = per_class_rows
            .iter()
            .enumerate()
            .max_by_key(|(_, rows)| rows.len())
            .map(|(class_index, _)| class_index)
            .expect("at least one class");
        per_class_validation[largest_class] = 1;
    }
    let fold_seed = sk_derive_seed(global_seed, SK_RANDOM_DOMAIN_FOLD, 0);
    let mut validation_indices = Vec::new();
    per_class_rows
        .iter()
        .zip(per_class_validation.iter())
        .enumerate()
        .filter(|(_, (_, count))| **count > 0)
        .for_each(|(class_index, (rows, &count))| {
            let class_seed = sk_derive_seed(fold_seed, SK_RANDOM_DOMAIN_TREE, class_index as u64);
            let local_positions = SKRandomGenerator::new(class_seed)
                .sample_indices_without_replacement(rows.len(), count);
            validation_indices.extend(local_positions.into_iter().map(|position| rows[position]));
        });
    validation_indices.sort_unstable();
    let training_indices = sk_complement_indices(row_count, &validation_indices);
    Ok((training_indices, validation_indices))
}

fn sk_complement_indices(row_count: usize, sorted_selected: &[usize]) -> Vec<usize> {
    (0..row_count)
        .filter(|row| sorted_selected.binary_search(row).is_err())
        .collect()
}
