//! Fit and transformation contracts (spec `estimator-contracts`).
//!
//! Two fit traits separated by the supervision axis. Fit returns a **distinct
//! model type** (sole bearer of the learned state), operates on a **shared
//! reference** of the configured estimator, and produces models that are
//! `Send + Sync` by construction. Prediction exists only on the model type, so
//! predict-without-fit is unrepresentable. The transformer declares its output
//! as an associated type for static pipeline chaining.

use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;
use crate::target_view::SKTargetView;

/// Unsupervised fit: receives only features; returns a distinct model type.
///
/// `fit` takes `&self`, so the configured estimator stays unchanged and
/// reusable for new fits (including concurrent ones). The feature input `X`
/// converts (zero-copy) into a [`SKDataView`] through the fallible seam; its
/// conversion error flows into the operation's `Result`.
pub trait SKUnsupervisedFit<F: SKFloat> {
    /// The fitted model type — the sole bearer of the learned state.
    type Model;
    /// The operation error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Fit on a shared reference, returning the model.
    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}

/// Supervised fit: requires features and targets.
pub trait SKSupervisedFit<F: SKFloat> {
    /// The fitted model type.
    type Model;
    /// The operation error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Fit on a shared reference with features and targets.
    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a>, Error = SKError>;
}

/// A feature transformer whose output type is declared as an associated type,
/// enabling statically validated pipeline chaining.
pub trait SKFeatureTransformer<F: SKFloat> {
    /// The type produced by the transformation.
    type Output;
    /// The operation error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Transform features into the associated output type.
    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}

/// Prediction lives on the fitted model type (never on the configured
/// estimator). Required by the convenient scoring forms, which run inference.
pub trait SKPredictor<F: SKFloat> {
    /// The inference error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Predict continuous scores/class indices for the given features.
    fn predict<'a, X>(&self, features: X) -> Result<ndarray::Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}

#[cfg(test)]
mod fit_traits_tests;
