use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState},
};

use crate::robust_scaler::core_implementation::SKRobustScaler;

pub struct SKRobustScalerBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) with_centering: bool,
    pub(super) with_scaling: bool,
    pub(super) quantile_low: F,
    pub(super) quantile_high: F,
    pub(super) unit_variance: bool,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKRobustScalerBuilder<F> {
    pub fn new() -> SKRobustScalerBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            with_centering: true,
            with_scaling: true,
            quantile_low: F::from(25.0).expect("25 fits in scalar"),
            quantile_high: F::from(75.0).expect("75 fits in scalar"),
            unit_variance: false,
            backend: SKBackendKind::Faer,
        }
    }

    /// Center on the median when true (default), else leave the values centered
    /// at zero.
    pub fn with_centering(mut self, value: bool) -> Self {
        self.with_centering = value;
        self
    }

    /// Scale by the interquantile range when true (default), else leave a unit
    /// scale.
    pub fn with_scaling(mut self, value: bool) -> Self {
        self.with_scaling = value;
        self
    }

    /// The `(low, high)` percentile pair the scale is built from; defaults to
    /// `(25, 75)`.
    pub fn quantile_range(mut self, range: (F, F)) -> Self {
        self.quantile_low = range.0;
        self.quantile_high = range.1;
        self
    }

    /// Divide the scale by the standard-normal interquantile span so normally
    /// distributed features land on unit variance; defaults to `false`.
    pub fn unit_variance(mut self, value: bool) -> Self {
        self.unit_variance = value;
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

impl<F: SKFloat> Default for SKRobustScalerBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKRobustScaler<F>> for SKRobustScalerBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKRobustScaler<F>, SKError> {
        Ok(SKRobustScaler {
            plan: self.state,
            with_centering: self.with_centering,
            with_scaling: self.with_scaling,
            quantile_low: self.quantile_low,
            quantile_high: self.quantile_high,
            unit_variance: self.unit_variance,
            backend: self.backend,
        })
    }
}
