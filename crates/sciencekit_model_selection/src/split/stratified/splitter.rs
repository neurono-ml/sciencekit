//! The stratified K-fold splitter.

use ndarray::ArrayView1;
use sciencekit_common::{
    SKDataView, SKError, SKExecutionMode, SKFloat, SKRandomGenerator, SKTargetView,
    SK_RANDOM_DOMAIN_FOLD, sk_derive_seed,
};

use super::builder::SKStratifiedKFoldBuilder;
use super::configuration::SKStratifiedKFoldConfiguration;
use crate::{SKIndexSplit, SKSplitter};

use crate::split::validation::{
    sk_class_groups, sk_number_of_rows, sk_split_from_test, sk_validate_cardinality,
    sk_validate_non_empty, sk_validate_number_of_splits,
};

/// Stratified K-fold, constructed only through [`SKStratifiedKFoldBuilder`].
///
/// Rows are grouped by class and dealt round-robin into the folds, so every
/// fold's test partition contains every class in proportion to the data. A
/// class with fewer members than the fold count is rejected instead of silently
/// dropped. Continuous targets are rejected: stratification requires
/// categorical labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKStratifiedKFold {
    configuration: SKStratifiedKFoldConfiguration,
}

impl SKStratifiedKFold {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKStratifiedKFoldBuilder {
        SKStratifiedKFoldBuilder::new()
    }

    /// The configured number of folds.
    pub fn number_of_splits(&self) -> usize {
        self.configuration.number_of_splits
    }

    /// Whether a seeded shuffle precedes dealing.
    pub fn shuffle(&self) -> bool {
        self.configuration.shuffle
    }

    /// The deterministic seed.
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// The configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }

    pub(super) fn from_configuration(configuration: SKStratifiedKFoldConfiguration) -> Self {
        Self { configuration }
    }

    fn ordered_class_members(&self, class_index: usize, members: &[usize]) -> Vec<usize> {
        if !self.configuration.shuffle {
            return members.to_vec();
        }
        let seed = sk_derive_seed(
            self.configuration.random_seed,
            SK_RANDOM_DOMAIN_FOLD,
            class_index as u64,
        );
        let mut generator = SKRandomGenerator::new(seed);
        let mut ordered = members.to_vec();
        generator.shuffle(&mut ordered);
        ordered
    }
}

impl<F: SKFloat> SKSplitter<F> for SKStratifiedKFold {
    fn split(
        &self,
        features: SKDataView<'_, F>,
        targets: Option<SKTargetView<'_, F>>,
        _groups: Option<ArrayView1<'_, i64>>,
    ) -> Result<Vec<SKIndexSplit>, SKError> {
        let number_of_rows = sk_number_of_rows(&features);
        sk_validate_non_empty(number_of_rows, "SKStratifiedKFold::split")?;
        let number_of_splits = self.configuration.number_of_splits;
        sk_validate_number_of_splits(number_of_splits, number_of_rows)?;

        let class_groups = sk_class_groups(targets)?;
        for members in &class_groups {
            sk_validate_cardinality("targets", members.len(), number_of_splits)?;
        }

        let mut test_folds: Vec<Vec<usize>> = vec![Vec::new(); number_of_splits];
        for (class_index, members) in class_groups.iter().enumerate() {
            let ordered = self.ordered_class_members(class_index, members);
            for (position, &row) in ordered.iter().enumerate() {
                test_folds[position % number_of_splits].push(row);
            }
        }

        Ok(test_folds
            .into_iter()
            .map(|test| sk_split_from_test(number_of_rows, test))
            .collect())
    }
}
