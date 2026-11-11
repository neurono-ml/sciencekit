//! Mandatory builder for [`SKIncrementalPCA`] (spec `incremental-pca`).

use std::marker::PhantomData;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, builders::SKBuilder,
    sk_validate_hyperparameter,
};

use super::config::SKIncrementalPCAConfiguration;
use super::core_implementation::SKIncrementalPCA;

/// Builder-only construction surface for [`SKIncrementalPCA`].
#[derive(Debug, Clone, Copy)]
pub struct SKIncrementalPCABuilder<F: SKFloat> {
    pub(super) configuration: SKIncrementalPCAConfiguration,
    scalar: PhantomData<fn() -> F>,
}

impl<F: SKFloat> SKIncrementalPCABuilder<F> {
    /// Creates a builder with the documented defaults: `number_of_components`
    /// unset (`None`, meaning the feature count), `batch_size = 1024`,
    /// `whiten = false`, `maximum_number_of_epochs = 1`, `Faer` backend and
    /// `Automatic` execution mode.
    pub fn new() -> Self {
        Self {
            configuration: SKIncrementalPCAConfiguration::default_configuration(),
            scalar: PhantomData,
        }
    }

    /// Sets the number of principal components to retain (must be strictly
    /// positive).
    pub fn number_of_components(mut self, value: usize) -> Self {
        self.configuration.number_of_components = Some(value);
        self
    }

    /// Sets the row block size used when `fit` chunks an in-memory matrix (must
    /// be strictly positive).
    pub fn batch_size(mut self, value: usize) -> Self {
        self.configuration.batch_size = value;
        self
    }

    /// Enables or disables unit-variance scaling of transformed components.
    pub fn whiten(mut self, value: bool) -> Self {
        self.configuration.whiten = value;
        self
    }

    /// Sets the epoch budget for reopenable streaming providers (must be
    /// strictly positive).
    pub fn maximum_number_of_epochs(mut self, value: usize) -> Self {
        self.configuration.maximum_number_of_epochs = value;
        self
    }

    /// Sets the math backend recorded in operation spans and dispatched per
    /// operation.
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.configuration.backend = value;
        self
    }

    /// Sets the execution intent (`Automatic` by default).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.builder_state.execution_mode(mode);
        self
    }

    fn validated_configuration(&self) -> Result<SKIncrementalPCAConfiguration, SKError> {
        if let Some(components) = self.configuration.number_of_components {
            sk_validate_hyperparameter(
                "number_of_components",
                components > 0,
                "must be strictly positive",
            )?;
        }
        sk_validate_hyperparameter(
            "batch_size",
            self.configuration.batch_size > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "maximum_number_of_epochs",
            self.configuration.maximum_number_of_epochs > 0,
            "must be strictly positive",
        )?;
        Ok(self.configuration)
    }
}

impl<F: SKFloat> Default for SKIncrementalPCABuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKIncrementalPCA<F>> for SKIncrementalPCABuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.builder_state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKIncrementalPCA<F>, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKIncrementalPCA::from_configuration(configuration))
    }
}
