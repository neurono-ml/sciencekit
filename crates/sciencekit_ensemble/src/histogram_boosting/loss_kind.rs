//! Public loss selector for [`super::SKHistGradientBoostingRegressor`].

/// Regression loss selector for [`super::SKHistGradientBoostingRegressor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKHistGradientBoostingRegressionLoss {
    /// Half squared-error loss `0.5 * (y - f)²` (default).
    #[default]
    SquaredError,
    /// Absolute-error loss `|y - f|` with unit Hessian surrogate.
    AbsoluteError,
}
