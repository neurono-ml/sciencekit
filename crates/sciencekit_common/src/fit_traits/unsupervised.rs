//! Unsupervised fit contract.

use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;

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
