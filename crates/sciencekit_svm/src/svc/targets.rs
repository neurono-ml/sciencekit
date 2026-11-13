//! Classification target canonicalization for `SKSVC`
//! (spec `support-vector-machines`).

use sciencekit_common::{SKError, SKFloat, SKLabelTable, SKTargetView, sk_canonicalize_labels};

/// Canonicalize integer or nominal labels and ensure at least two distinct
/// classes are present.
pub(crate) fn sk_canonicalize_svc_targets<F: SKFloat>(
    targets: &SKTargetView<'_, F>,
    expected_rows: usize,
) -> Result<(Vec<usize>, SKLabelTable), SKError> {
    let (class_indices, classes) = match targets {
        SKTargetView::Nominal(labels) => sk_canonicalize_labels(labels),
        SKTargetView::Integer(values) => {
            let owned: Vec<String> = values.iter().map(|value| value.to_string()).collect();
            let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
            sk_canonicalize_labels(&borrowed)
        }
        SKTargetView::Continuous(_) => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "continuous",
                suggestion: "pass integer or nominal class labels to SKSVC",
            });
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "pass integer or nominal class labels to SKSVC",
            });
        }
    };
    if class_indices.len() != expected_rows {
        return Err(SKError::shape_mismatch_2d(
            expected_rows,
            1,
            class_indices.len(),
            1,
        ));
    }
    if classes.number_of_classes() < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "classes",
            reason: "SKSVC requires at least two distinct classes".to_owned(),
        });
    }
    Ok((class_indices, classes))
}
