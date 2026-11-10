//! The group K-fold splitter.

use std::collections::BTreeMap;

use ndarray::ArrayView1;
use sciencekit_common::{
    SKDataView, SKError, SKExecutionMode, SKFloat, SKRandomGenerator, SKTargetView,
};

use super::builder::SKGroupKFoldBuilder;
use super::configuration::SKGroupKFoldConfiguration;
use crate::{SKIndexSplit, SKSplitter};

use crate::split::validation::{
    sk_number_of_rows, sk_split_from_test, sk_validate_cardinality, sk_validate_finite_features,
    sk_validate_finite_targets, sk_validate_non_empty, sk_validate_number_of_splits,
};

/// Group K-fold, constructed only through [`SKGroupKFoldBuilder`].
///
/// Distinct groups are shuffled with the seed and dealt round-robin into the
/// folds; every row of a group goes to the same fold, so no group ever appears on
/// both the train and test side of a fold. The split count defaults to the number
/// of distinct groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKGroupKFold {
    configuration: SKGroupKFoldConfiguration,
}

impl SKGroupKFold {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKGroupKFoldBuilder {
        SKGroupKFoldBuilder::new()
    }

    /// The explicit split count, or `None` when it follows the group count.
    pub fn number_of_splits(&self) -> Option<usize> {
        self.configuration.number_of_splits
    }

    /// Whether the distinct groups are seeded-shuffled.
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

    pub(super) fn from_configuration(configuration: SKGroupKFoldConfiguration) -> Self {
        Self { configuration }
    }
}

impl<F: SKFloat> SKSplitter<F> for SKGroupKFold {
    fn split(
        &self,
        features: SKDataView<'_, F>,
        _targets: Option<SKTargetView<'_, F>>,
        groups: Option<ArrayView1<'_, i64>>,
    ) -> Result<Vec<SKIndexSplit>, SKError> {
        let number_of_rows = sk_number_of_rows(&features);
        sk_validate_non_empty(number_of_rows, "SKGroupKFold::split")?;
        sk_validate_finite_features(&features)?;
        sk_validate_finite_targets(_targets)?;
        let groups = groups.ok_or_else(|| SKError::InvalidHyperparameter {
            name: "groups",
            reason: "group K-fold requires group labels".to_owned(),
        })?;
        if groups.len() != number_of_rows {
            return Err(SKError::ShapeMismatch {
                expected: vec![number_of_rows],
                found: vec![groups.len()],
            });
        }

        let mut groups_by_value: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (row, &group) in groups.iter().enumerate() {
            groups_by_value.entry(group).or_default().push(row);
        }
        let number_of_groups = groups_by_value.len();
        let number_of_splits = self
            .configuration
            .number_of_splits
            .unwrap_or(number_of_groups);
        sk_validate_number_of_splits(number_of_splits, number_of_rows)?;
        sk_validate_cardinality("number_of_splits", number_of_groups, number_of_splits)?;

        let mut group_order: Vec<i64> = groups_by_value.keys().copied().collect();
        if self.configuration.shuffle {
            let mut generator = SKRandomGenerator::new(self.configuration.random_seed);
            generator.shuffle(&mut group_order);
        }

        let mut test_folds: Vec<Vec<usize>> = vec![Vec::new(); number_of_splits];
        for (position, group) in group_order.iter().enumerate() {
            let rows = groups_by_value
                .get(group)
                .expect("the order comes from the group map");
            test_folds[position % number_of_splits].extend_from_slice(rows);
        }

        Ok(test_folds
            .into_iter()
            .map(|mut test| {
                test.sort_unstable();
                sk_split_from_test(number_of_rows, test)
            })
            .collect())
    }
}
