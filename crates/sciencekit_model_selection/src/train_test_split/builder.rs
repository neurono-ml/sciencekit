//! The mandatory train/test split builder.

use sciencekit_common::{
    SKBuilder, SKBuilderState, SKError, SKExecutionMode, sk_validate_hyperparameter,
};

use super::configuration::SKTrainTestSplitConfiguration;
use super::partition_size::SKPartitionSize;
use super::splitter::SKTrainTestSplitter;

/// Builder-only construction surface for [`SKTrainTestSplitter`].
#[derive(Debug, Clone)]
pub struct SKTrainTestSplitBuilder {
    pub(super) state: SKBuilderState,
    pub(super) test_size: Option<SKPartitionSize>,
    pub(super) train_size: Option<SKPartitionSize>,
    pub(super) shuffle: bool,
    pub(super) random_seed: u64,
    pub(super) stratify: bool,
}

impl SKTrainTestSplitBuilder {
    /// A builder carrying the documented defaults: a `0.25` test partition when
    /// neither size is set, shuffling enabled, seed `0`, no stratification and
    /// the `Automatic` execution mode.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            test_size: None,
            train_size: None,
            shuffle: true,
            random_seed: 0,
            stratify: false,
        }
    }

    /// Sets the test partition size by fraction or count.
    pub fn test_size(mut self, value: SKPartitionSize) -> Self {
        self.test_size = Some(value);
        self
    }

    /// Sets the test partition as a fraction of the rows.
    pub fn test_fraction(mut self, value: f64) -> Self {
        self.test_size = Some(SKPartitionSize::Fraction(value));
        self
    }

    /// Sets the test partition as an explicit row count.
    pub fn test_count(mut self, value: usize) -> Self {
        self.test_size = Some(SKPartitionSize::Count(value));
        self
    }

    /// Sets the train partition size by fraction or count.
    pub fn train_size(mut self, value: SKPartitionSize) -> Self {
        self.train_size = Some(value);
        self
    }

    /// Sets the train partition as a fraction of the rows.
    pub fn train_fraction(mut self, value: f64) -> Self {
        self.train_size = Some(SKPartitionSize::Fraction(value));
        self
    }

    /// Sets the train partition as an explicit row count.
    pub fn train_count(mut self, value: usize) -> Self {
        self.train_size = Some(SKPartitionSize::Count(value));
        self
    }

    /// Enables or disables seeded shuffling before partitioning.
    pub fn shuffle(mut self, value: bool) -> Self {
        self.shuffle = value;
        self
    }

    /// Sets the deterministic seed used when shuffling is enabled.
    pub fn random_seed(mut self, value: u64) -> Self {
        self.random_seed = value;
        self
    }

    /// Enables class-proportion-preserving stratification; requires categorical
    /// targets at split time.
    pub fn stratify(mut self, value: bool) -> Self {
        self.stratify = value;
        self
    }

    /// Sets the execution intent (`Automatic` by default).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn validated_configuration(&self) -> Result<SKTrainTestSplitConfiguration, SKError> {
        if let Some(test_size) = self.test_size {
            sk_validate_hyperparameter(
                "test_size",
                test_size.is_valid(),
                "must be a fraction in (0, 1) or a strictly positive count",
            )?;
        }
        if let Some(train_size) = self.train_size {
            sk_validate_hyperparameter(
                "train_size",
                train_size.is_valid(),
                "must be a fraction in (0, 1) or a strictly positive count",
            )?;
        }
        if let (
            Some(SKPartitionSize::Fraction(test_fraction)),
            Some(SKPartitionSize::Fraction(train_fraction)),
        ) = (self.test_size, self.train_size)
        {
            sk_validate_hyperparameter(
                "train_size",
                test_fraction + train_fraction < 1.0,
                "the test and train fractions must leave at least one row unassigned",
            )?;
        }
        Ok(SKTrainTestSplitConfiguration {
            test_size: self.test_size,
            train_size: self.train_size,
            shuffle: self.shuffle,
            random_seed: self.random_seed,
            stratify: self.stratify,
            execution_mode: self.state.execution_intent(),
        })
    }
}

impl Default for SKTrainTestSplitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKTrainTestSplitter> for SKTrainTestSplitBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKTrainTestSplitter, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKTrainTestSplitter::from_configuration(configuration))
    }
}
