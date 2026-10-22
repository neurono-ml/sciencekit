//! Target canonicalization for the logistic estimators (spec
//! `logistic-regression`).
//!
//! Integer and nominal targets are mapped to compact canonical indices in
//! first-occurrence order through [`SKLabelTable`]; continuous targets are
//! rejected.

use sciencekit_common::{SKError, SKLabelTable, SKTargetView, sk_canonicalize_labels};

/// Canonicalize a target view into compact class indices and the label table.
///
/// Integer targets are canonicalized by their decimal representation, nominal
/// targets by their text; a continuous target view is rejected with a
/// structured error.
pub(crate) fn sk_canonicalize_targets(
    targets: SKTargetView<'_, impl sciencekit_common::SKFloat>,
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

/// Canonicalize a target view against an existing class table, rejecting a
/// label the fit never saw.
pub(crate) fn sk_canonicalize_against(
    classes: &SKLabelTable,
    targets: SKTargetView<'_, impl sciencekit_common::SKFloat>,
) -> Result<Vec<usize>, SKError> {
    let lookup = |label: &str| {
        classes
            .index_of(label)
            .ok_or_else(|| SKError::InvalidHyperparameter {
                name: "targets",
                reason: format!("label `{label}` was not present at fit time"),
            })
    };
    match targets {
        SKTargetView::Nominal(labels) => labels.iter().map(|label| lookup(label)).collect(),
        SKTargetView::Integer(values) => values
            .iter()
            .map(|value| lookup(&value.to_string()))
            .collect(),
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
