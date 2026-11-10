//! Fitting the final supervised estimator on the last block.
//!
//! The shared fit contracts take features and targets under one lifetime, but a
//! pipeline's last block is a local value while the targets borrow the caller's
//! data. This helper materialises the targets into a local owned buffer of the
//! same kind, so the final fit borrows only locals and the two lifetimes meet.

use sciencekit_common::{SKError, SKFloat, SKSupervisedFit, SKTargetView};

use crate::SKFeatureBlock;

/// Fit the final estimator on `block` with a local copy of the targets.
pub(super) fn sk_fit_final<F, Estimator>(
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
        SKTargetView::Integer(view) => {
            let owned = view.to_owned();
            estimator.fit(block, owned.view()).map_err(Into::into)
        }
        SKTargetView::Continuous(view) => {
            let owned = view.to_owned();
            estimator.fit(block, owned.view()).map_err(Into::into)
        }
        SKTargetView::Nominal(text) => {
            let owned: Vec<String> = text.iter().map(|label| label.to_string()).collect();
            let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
            estimator.fit(block, borrowed.as_slice()).map_err(Into::into)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the targets to integers, continuous values or text",
        }),
    }
}
