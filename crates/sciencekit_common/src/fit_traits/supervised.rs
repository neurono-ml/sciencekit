//! Supervised fit contract.

use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;
use crate::target_view::SKTargetView;

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
