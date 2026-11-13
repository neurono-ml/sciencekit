//! Shape and finiteness validation shared by the adapter fits.

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use crate::logistic_regression::validation::unsupported_representation;

/// Reject any non-finite feature, naming the first offending position.
pub(super) fn sk_validate_finite_features<F: SKFloat>(
    data: &SKDataView<'_, F>,
) -> Result<(), SKError> {
    let finite = match data {
        SKDataView::Dense(dense) => dense.iter().all(|value| value.is_finite()),
        SKDataView::Sparse(sparse) => sparse.data().iter().all(|value| value.is_finite()),
        _ => return Err(unsupported_representation()),
    };
    if finite {
        Ok(())
    } else {
        Err(SKError::NonFiniteValue { position: 0 })
    }
}

/// Reject any non-finite continuous target, naming its position.
pub(super) fn sk_validate_finite_targets<F: SKFloat>(
    targets: ArrayView1<'_, F>,
) -> Result<(), SKError> {
    match targets.iter().position(|value| !value.is_finite()) {
        Some(position) => Err(SKError::NonFiniteValue { position }),
        None => Ok(()),
    }
}
