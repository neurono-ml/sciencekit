//! Fitting the final supervised estimator on the last block.
//!
//! The shared fit contracts take features and targets under one lifetime, but a
//! pipeline's last block is a local value while the targets borrow the caller's
//! data. Dispatching on the concrete target view lets both borrows shorten to
//! the local intersection, so no target copy is needed.

use sciencekit_common::{SKError, SKFloat, SKSupervisedFit, SKTargetView};

use crate::SKFeatureBlock;

/// Fit the final estimator on `block` with the borrowed targets.
pub(crate) fn sk_fit_final<F, Estimator>(
    estimator: &Estimator,
    block: &SKFeatureBlock<F>,
    targets: SKTargetView<'_, F>,
) -> Result<Estimator::Model, SKError>
where
    F: SKFloat,
    Estimator: SKSupervisedFit<F>,
    Estimator::Error: Into<SKError>,
{
    match targets {
        SKTargetView::Integer(view) => estimator.fit(block, view).map_err(Into::into),
        SKTargetView::Continuous(view) => estimator.fit(block, view).map_err(Into::into),
        SKTargetView::Nominal(text) => estimator.fit(block, text).map_err(Into::into),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the targets to integers, continuous values or text",
        }),
    }
}
