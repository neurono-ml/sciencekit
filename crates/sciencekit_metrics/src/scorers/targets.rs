//! Target-view conversion shared by the scorer adapters.

use ndarray::Array1;
use sciencekit_common::{SKError, SKFloat, SKTargetView};

/// Convert a target view into canonical `i64` labels.
///
/// Integers pass through; continuous values must be finite and are rounded to
/// the nearest integer; nominal targets are rejected with guidance to encode
/// them first.
pub(super) fn sk_labels_from_targets<F: SKFloat>(
    targets: &SKTargetView<'_, F>,
) -> Result<Array1<i64>, SKError> {
    match targets {
        SKTargetView::Integer(view) => Ok((*view).to_owned()),
        SKTargetView::Continuous(view) => {
            let mut labels = Vec::with_capacity(view.len());
            for (position, value) in view.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SKError::NonFiniteValue { position });
                }
                let label = num_traits::ToPrimitive::to_i64(&value.round()).ok_or_else(|| {
                    SKError::Conversion(
                        "a continuous label outside the i64 range cannot be represented as i64"
                            .to_string(),
                    )
                })?;
                labels.push(label);
            }
            Ok(Array1::from_vec(labels))
        }
        SKTargetView::Nominal(_) => Err(SKError::UnsupportedRepresentation {
            representation: "nominal",
            suggestion: "encode nominal targets to indices before scoring",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the target representation before scoring",
        }),
    }
}
