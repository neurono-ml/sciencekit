//! Class-target canonicalization for the neighbor classifier
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::{
    SKError, SKFloat, SKLabelTable, SKTargetView, sk_canonicalize_labels,
};

/// Canonicalize integer or nominal class labels, rejecting continuous targets.
///
/// Integer labels are rendered as their decimal text so the shared label table
/// keeps a reversible canonical order; predictions are returned as its indices.
pub(crate) fn sk_canonicalize_classes<F: SKFloat>(
    targets: &SKTargetView<'_, F>,
) -> Result<(Vec<usize>, SKLabelTable), SKError> {
    match targets {
        SKTargetView::Nominal(labels) => Ok(sk_canonicalize_labels(labels)),
        SKTargetView::Integer(values) => {
            let owned: Vec<String> = values.iter().map(|value| value.to_string()).collect();
            let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
            Ok(sk_canonicalize_labels(&borrowed))
        }
        SKTargetView::Continuous(_) => Err(SKError::UnsupportedRepresentation {
            representation: "continuous",
            suggestion: "pass integer or nominal class labels to a classifier",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "pass integer or nominal class labels to a classifier",
        }),
    }
}
