//! Feature-transformer contract.

use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;

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
