//! Outlier prediction contract (spec `outlier-prediction`).
//!
//! Fitted outlier detectors return one one-byte [`SKOutlierLabel`] per input
//! row (`Inlier` or `Outlier`), independent of the feature scalar, and expose
//! both [`SKOutlierPredictor::score_samples`] (where larger values mean more
//! normal) and [`SKOutlierPredictor::decision_function`] (`score_samples -
//! offset`, negative for outliers and non-negative for inliers). Integer `-1`
//! and `1` exist only at the explicit interoperability boundary.

use ndarray::Array1;

use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;

/// Typed label emitted by an outlier detector for each observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKOutlierLabel {
    /// Normal observation (`1` at the external interop boundary).
    Inlier,
    /// Anomalous observation (`-1` at the external interop boundary).
    Outlier,
}

impl SKOutlierLabel {
    /// Reports whether this label is the [`Self::Outlier`] variant.
    pub fn is_outlier(self) -> bool {
        matches!(self, Self::Outlier)
    }

    /// Reports whether this label is the [`Self::Inlier`] variant.
    pub fn is_inlier(self) -> bool {
        matches!(self, Self::Inlier)
    }

    /// Converts to the external `i64` convention (`Inlier -> 1`, `Outlier -> -1`).
    pub fn to_i64(self) -> i64 {
        match self {
            Self::Inlier => 1,
            Self::Outlier => -1,
        }
    }

    /// Converts from the external `i64` convention (`1 -> Inlier`, `-1 -> Outlier`).
    ///
    /// Any other integer returns a structured [`SKError::Conversion`].
    pub fn from_i64(value: i64) -> Result<Self, SKError> {
        match value {
            1 => Ok(Self::Inlier),
            -1 => Ok(Self::Outlier),
            _ => Err(SKError::Conversion(format!(
                "outlier label {value} must be 1 (Inlier) or -1 (Outlier)"
            ))),
        }
    }
}

/// Model-side outlier prediction contract.
pub trait SKOutlierPredictor<F: SKFloat> {
    /// The inference error type; converts from the central [`SKError`].
    type Error: From<SKError>;

    /// Threshold offset subtracted from [`Self::score_samples`] to form [`Self::decision_function`].
    fn offset(&self) -> F;

    /// Normality score per input row (larger values indicate more normal observations).
    fn score_samples<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;

    /// Shifted normality score (`score_samples - offset`) per input row.
    ///
    /// Negative values indicate [`SKOutlierLabel::Outlier`]; non-negative values
    /// indicate [`SKOutlierLabel::Inlier`].
    fn decision_function<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;

    /// Predicts one typed [`SKOutlierLabel`] per input row.
    fn predict<'a, X>(&self, features: X) -> Result<Array1<SKOutlierLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}
