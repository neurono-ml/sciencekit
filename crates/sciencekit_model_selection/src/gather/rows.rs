//! The public row-gathering entry point.

use sciencekit_common::{SKDataView, SKError, SKFeatureBlock, SKFloat, SKTargetView};

use super::{SKTargetPartition, sk_gather_feature_rows, sk_gather_target_rows};

/// Gathers the selected rows of features and optional targets into owned data.
///
/// A dense input yields a dense block and a sparse input yields a CSR block with
/// only the stored entries copied — a sparse input is never densified. Target
/// gathering preserves the target representation. The output rows follow
/// `row_indices` order, so a caller controls the partition order.
pub fn sk_gather_rows<F: SKFloat>(
    features: SKDataView<'_, F>,
    targets: Option<SKTargetView<'_, F>>,
    row_indices: &[usize],
) -> Result<(SKFeatureBlock<F>, Option<SKTargetPartition<F>>), SKError> {
    let number_of_rows = match features {
        SKDataView::Dense(view) => view.nrows(),
        SKDataView::Sparse(view) => view.rows(),
        _ => 0,
    };
    if let Some(targets) = targets {
        if targets.len() != number_of_rows {
            return Err(SKError::ShapeMismatch {
                expected: vec![number_of_rows],
                found: vec![targets.len()],
            });
        }
    }
    let gathered_features = sk_gather_feature_rows(features, row_indices)?;
    let gathered_targets = match targets {
        Some(targets) => Some(sk_gather_target_rows(targets, row_indices)?),
        None => None,
    };
    Ok((gathered_features, gathered_targets))
}
