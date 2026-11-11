//! Error composition for generic bagging estimators.

use std::error::Error;
use std::fmt;

use sciencekit_common::SKError;

/// Newtype wrapper marking an error `E` as originating from the base estimator
/// so `From<SKBaseEstimatorError<E>>` coexists with `From<SKError>` even when
/// `E = SKError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKBaseEstimatorError<E>(pub E);

/// Error returned by generic bagging `fit` and prediction operations.
///
/// Preserves the base estimator's concrete error type `E` alongside central
/// [`SKError`] validation and shape failures.
#[derive(Debug)]
pub enum SKBaggingError<E> {
    /// A central validation, shape or execution error owned by bagging.
    Central(SKError),
    /// An error raised by the underlying base estimator or fitted base model.
    Base(E),
}

impl<E> SKBaggingError<E> {
    /// Wrap a base-estimator error in [`SKBaggingError::Base`].
    pub fn from_base(error: E) -> Self {
        Self::Base(error)
    }

    /// Borrow the central error when this is [`SKBaggingError::Central`].
    pub fn as_central(&self) -> Option<&SKError> {
        match self {
            Self::Central(error) => Some(error),
            Self::Base(_) => None,
        }
    }

    /// Borrow the base error when this is [`SKBaggingError::Base`].
    pub fn as_base(&self) -> Option<&E> {
        match self {
            Self::Central(_) => None,
            Self::Base(error) => Some(error),
        }
    }
}

impl<E> From<SKError> for SKBaggingError<E> {
    fn from(error: SKError) -> Self {
        Self::Central(error)
    }
}

impl<E> From<SKBaseEstimatorError<E>> for SKBaggingError<E> {
    fn from(wrapper: SKBaseEstimatorError<E>) -> Self {
        Self::Base(wrapper.0)
    }
}

impl From<SKBaggingError<SKError>> for SKError {
    fn from(error: SKBaggingError<SKError>) -> Self {
        match error {
            SKBaggingError::Central(central) | SKBaggingError::Base(central) => central,
        }
    }
}

impl<E: fmt::Display> fmt::Display for SKBaggingError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Central(error) => fmt::Display::fmt(error, formatter),
            Self::Base(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<E: Error + 'static> Error for SKBaggingError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Central(error) => Some(error),
            Self::Base(error) => Some(error),
        }
    }
}
