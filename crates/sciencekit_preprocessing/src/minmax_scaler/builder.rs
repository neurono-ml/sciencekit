use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState},
};

use crate::minmax_scaler::core_implementation::SKMinMaxScaler;

pub struct SKMinMaxScalerBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) feature_minimum: F,
    pub(super) feature_maximum: F,
    pub(super) clip: bool,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKMinMaxScalerBuilder<F> {
    pub fn new() -> SKMinMaxScalerBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            feature_minimum: F::zero(),
            feature_maximum: F::one(),
            clip: false,
            backend: SKBackendKind::Faer,
        }
    }

    /// Set the target range transformed features are mapped onto; defaults to
    /// `(0, 1)`.
    pub fn feature_range(mut self, range: (F, F)) -> Self {
        self.feature_minimum = range.0;
        self.feature_maximum = range.1;
        self
    }

    /// Clamp transformed values to the target range; defaults to `false`.
    pub fn clip(mut self, value: bool) -> Self {
        self.clip = value;
        self
    }

    /// The backend kind carried into every operation span and the fitted model;
    /// defaults to [`SKBackendKind::Faer`].
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

impl<F: SKFloat> Default for SKMinMaxScalerBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKMinMaxScaler<F>> for SKMinMaxScalerBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKMinMaxScaler<F>, SKError> {
        Ok(SKMinMaxScaler {
            plan: self.state,
            feature_minimum: self.feature_minimum,
            feature_maximum: self.feature_maximum,
            clip: self.clip,
            backend: self.backend,
        })
    }
}
