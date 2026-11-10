//! Index computation for the train/test splitter.

use ndarray::ArrayView1;
use sciencekit_common::{
    SK_RANDOM_DOMAIN_FOLD, SKDataView, SKError, SKFloat, SKRandomGenerator, SKTargetView,
    sk_derive_seed,
};

use super::core_implementation::SKTrainTestSplitter;
use crate::train_test_split::stratification::sk_allocate_class_counts;
use crate::{SKIndexSplit, SKSplitter};

use crate::split::validation::{
    sk_class_groups, sk_number_of_rows, sk_validate_finite_features, sk_validate_finite_targets,
    sk_validate_non_empty,
};

impl SKTrainTestSplitter {
    pub(super) fn split_indices<F: SKFloat>(
        &self,
        features: SKDataView<'_, F>,
        targets: Option<SKTargetView<'_, F>>,
    ) -> Result<SKIndexSplit, SKError> {
        let number_of_rows = sk_number_of_rows(&features);
        sk_validate_non_empty(number_of_rows, "sk_train_test_split")?;
        sk_validate_finite_features(&features)?;
        sk_validate_finite_targets(targets)?;
        if let Some(targets) = targets {
            if targets.len() != number_of_rows {
                return Err(SKError::ShapeMismatch {
                    expected: vec![number_of_rows],
                    found: vec![targets.len()],
                });
            }
        }

        let (number_of_train_rows, number_of_test_rows) = self.resolved_sizes(number_of_rows)?;
        if self.configuration.stratify {
            return self.stratified_indices(
                number_of_rows,
                number_of_train_rows,
                number_of_test_rows,
                targets,
            );
        }

        let permutation = if self.configuration.shuffle {
            crate::sk_shuffle_indices(self.configuration.random_seed, number_of_rows)
        } else {
            (0..number_of_rows).collect()
        };
        let test = permutation[..number_of_test_rows].to_vec();
        let train =
            permutation[number_of_test_rows..number_of_test_rows + number_of_train_rows].to_vec();
        Ok(SKIndexSplit::new(train, test))
    }

    fn stratified_indices<F: SKFloat>(
        &self,
        number_of_rows: usize,
        number_of_train_rows: usize,
        number_of_test_rows: usize,
        targets: Option<SKTargetView<'_, F>>,
    ) -> Result<SKIndexSplit, SKError> {
        let class_groups = sk_class_groups(targets)?;
        let class_sizes: Vec<usize> = class_groups.iter().map(Vec::len).collect();
        let test_counts =
            sk_allocate_class_counts(&class_sizes, number_of_test_rows, number_of_rows);
        let remaining_sizes: Vec<usize> = class_sizes
            .iter()
            .zip(&test_counts)
            .map(|(size, test)| size - test)
            .collect();
        let remaining_rows: usize = remaining_sizes.iter().sum();
        let train_counts =
            sk_allocate_class_counts(&remaining_sizes, number_of_train_rows, remaining_rows);

        let mut train = Vec::with_capacity(number_of_train_rows);
        let mut test = Vec::with_capacity(number_of_test_rows);
        for (class_index, members) in class_groups.iter().enumerate() {
            let mut ordered = members.clone();
            if self.configuration.shuffle {
                let seed = sk_derive_seed(
                    self.configuration.random_seed,
                    SK_RANDOM_DOMAIN_FOLD,
                    class_index as u64,
                );
                SKRandomGenerator::new(seed).shuffle(&mut ordered);
            }
            let test_count = test_counts[class_index];
            let train_count = train_counts[class_index];
            test.extend_from_slice(&ordered[..test_count]);
            train.extend_from_slice(&ordered[test_count..test_count + train_count]);
        }
        Ok(SKIndexSplit::new(train, test))
    }
}

impl<F: SKFloat> SKSplitter<F> for SKTrainTestSplitter {
    fn split(
        &self,
        features: SKDataView<'_, F>,
        targets: Option<SKTargetView<'_, F>>,
        _groups: Option<ArrayView1<'_, i64>>,
    ) -> Result<Vec<SKIndexSplit>, SKError> {
        Ok(vec![self.split_indices(features, targets)?])
    }
}
