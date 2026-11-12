//! Input validation for the linear SVC (spec `linear-support-vector-classifier`).

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKLabelTable, SKTargetView, sk_canonicalize_labels,
};

use crate::kernel::{sk_svm_view_dimensions, sk_validate_svm_finiteness};

/// Validate training features: non-empty, finite, and in a supported
/// representation (dense or CSR).
pub(crate) fn sk_validate_linear_svc_training_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = sk_svm_view_dimensions(&features)?;
    if rows == 0 || columns == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    sk_validate_svm_finiteness(&features)?;
    Ok((rows, columns))
}

/// Canonicalize integer or nominal labels and require at least two classes.
pub(crate) fn sk_canonicalize_linear_svc_targets<F: SKFloat>(
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
                suggestion: "pass integer or nominal class labels to SKLinearSVC",
            });
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "pass integer or nominal class labels to SKLinearSVC",
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
            reason: "SKLinearSVC requires at least two distinct classes".to_owned(),
        });
    }
    Ok((class_indices, classes))
}
