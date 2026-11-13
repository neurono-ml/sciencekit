//! The mandatory group K-fold builder.

use sciencekit_common::{
    SKBuilder, SKBuilderState, SKError, SKExecutionMode, sk_validate_hyperparameter,
};

use super::configuration::SKGroupKFoldConfiguration;
use super::splitter::SKGroupKFold;

/// Builder-only construction surface for [`SKGroupKFold`].
#[derive(Debug, Clone)]
pub struct SKGroupKFoldBuilder {
    pub(super) state: SKBuilderState,
    pub(super) number_of_splits: Option<usize>,
    pub(super) shuffle: bool,
    pub(super) random_seed: u64,
}

impl SKGroupKFoldBuilder {
    /// A builder carrying the documented defaults: one split per distinct
    /// group, seeded shuffling enabled, seed `0` and the `Automatic` execution
    /// mode.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_splits: None,
            shuffle: true,
            random_seed: 0,
        }
    }

    /// Sets the number of folds explicitly; unset uses the distinct group
    /// count.
    pub fn number_of_splits(mut self, value: usize) -> Self {
        self.number_of_splits = Some(value);
        self
    }

    /// Enables or disables seeded shuffling of the distinct groups.
    pub fn shuffle(mut self, value: bool) -> Self {
        self.shuffle = value;
        self
    }

    /// Sets the deterministic seed used when shuffling is enabled.
    pub fn random_seed(mut self, value: u64) -> Self {
        self.random_seed = value;
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

    fn validated_configuration(&self) -> Result<SKGroupKFoldConfiguration, SKError> {
        if let Some(number_of_splits) = self.number_of_splits {
            sk_validate_hyperparameter(
                "number_of_splits",
                number_of_splits >= 2,
                "must be at least 2",
            )?;
        }
        Ok(SKGroupKFoldConfiguration {
            number_of_splits: self.number_of_splits,
            shuffle: self.shuffle,
            random_seed: self.random_seed,
            execution_mode: self.state.execution_intent(),
        })
    }
}

impl Default for SKGroupKFoldBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKGroupKFold> for SKGroupKFoldBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKGroupKFold, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKGroupKFold::from_configuration(configuration))
    }
}
