//! The mandatory stratified K-fold builder.

use sciencekit_common::{
    SKBuilder, SKBuilderState, SKError, SKExecutionMode, sk_validate_hyperparameter,
};

use super::configuration::SKStratifiedKFoldConfiguration;
use super::splitter::SKStratifiedKFold;

/// Builder-only construction surface for [`SKStratifiedKFold`].
#[derive(Debug, Clone)]
pub struct SKStratifiedKFoldBuilder {
    pub(super) state: SKBuilderState,
    pub(super) number_of_splits: usize,
    pub(super) shuffle: bool,
    pub(super) random_seed: u64,
}

impl SKStratifiedKFoldBuilder {
    /// A builder carrying the documented defaults: five splits, no shuffle,
    /// seed `0` and the `Automatic` execution mode.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_splits: 5,
            shuffle: false,
            random_seed: 0,
        }
    }

    /// Sets the number of folds (must be at least two).
    pub fn number_of_splits(mut self, value: usize) -> Self {
        self.number_of_splits = value;
        self
    }

    /// Enables or disables seeded shuffling within each class.
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

    fn validated_configuration(&self) -> Result<SKStratifiedKFoldConfiguration, SKError> {
        sk_validate_hyperparameter(
            "number_of_splits",
            self.number_of_splits >= 2,
            "must be at least 2",
        )?;
        Ok(SKStratifiedKFoldConfiguration {
            number_of_splits: self.number_of_splits,
            shuffle: self.shuffle,
            random_seed: self.random_seed,
            execution_mode: self.state.execution_intent(),
        })
    }
}

impl Default for SKStratifiedKFoldBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKStratifiedKFold> for SKStratifiedKFoldBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKStratifiedKFold, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKStratifiedKFold::from_configuration(configuration))
    }
}
