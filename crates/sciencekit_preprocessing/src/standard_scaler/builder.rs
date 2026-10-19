use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode,
    builders::{SKBuilder, SKBuilderState},
};

use crate::standard_scaler::core_implementation::SKStandardScaler;

pub struct SKStandardScalerBuilder {
    pub(super) state: SKBuilderState,
    pub(super) with_mean: bool,
    pub(super) with_std: bool,
    pub(super) backend: SKBackendKind,
}

impl SKStandardScalerBuilder {
    pub fn new() -> SKStandardScalerBuilder {
        Self {
            state: SKBuilderState::new(),
            with_mean: true,
            with_std: true,
            backend: SKBackendKind::Faer,
        }
    }

    pub fn with_mean(mut self, value: bool) -> Self {
        self.with_mean = value;
        self
    }

    pub fn with_std(mut self, value: bool) -> Self {
        self.with_std = value;
        self
    }

    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl Default for SKStandardScalerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKStandardScaler> for SKStandardScalerBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKStandardScaler, SKError> {
        let built = SKStandardScaler {
            plan: self.state,
            with_mean: self.with_mean,
            with_std: self.with_std,
            backend: self.backend,
        };
        Ok(built)
    }
}
