//! Public loss-family selectors for [`super::SKGradientBoostingClassifier`] and
//! [`super::SKGradientBoostingRegressor`].

/// Classification loss selector for [`super::SKGradientBoostingClassifier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKGradientBoostingClassificationLoss {
    /// Cross-entropy log-loss with logistic sigmoid (binary) or softmax
    /// (multiclass) link (default).
    #[default]
    LogLoss,
}

/// Regression loss selector for [`super::SKGradientBoostingRegressor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKGradientBoostingRegressionLoss {
    /// Half squared-error loss `0.5 * (y - f)²` (default).
    #[default]
    SquaredError,
    /// Absolute-error loss `|y - f|` with unit Hessian surrogate.
    AbsoluteError,
    /// Huber loss with configurable clipping threshold `huber_delta`.
    Huber,
}
