//! Seeded without-replacement row subsampling for stochastic gradient boosting.

use ndarray::Array1;
use sciencekit_common::{
    SK_RANDOM_DOMAIN_INITIALIZATION, SKDataView, SKError, SKFloat, SKRandomGenerator,
    sk_derive_seed,
};
use sciencekit_tree::{SKStructureConstraints, SKTreeStructure};

use crate::bootstrap::{sk_gather_dense_rows, sk_gather_sparse_rows};

use super::stage::sk_stage_tree_seed;

/// Draw the row subset fitted by `(stage_index, class_index)`.
///
/// When `subsample >= 1`, every training row `0..row_count` is returned in
/// order; otherwise `floor(subsample * row_count).clamp(1, row_count)` distinct
/// rows are drawn deterministically without replacement.
pub(crate) fn sk_select_stage_rows<F: SKFloat>(
    row_count: usize,
    subsample: F,
    global_seed: u64,
    stage_index: usize,
    class_index: usize,
) -> Vec<usize> {
    if row_count == 0 {
        return Vec::new();
    }
    if subsample >= F::one() {
        return (0..row_count).collect();
    }
    let scaled = subsample * F::from(row_count).expect("row count fits in scalar");
    let sample_count = scaled.to_usize().unwrap_or(1).clamp(1, row_count);
    let tree_seed = sk_stage_tree_seed(global_seed, stage_index, class_index);
    let subsample_seed = sk_derive_seed(tree_seed, SK_RANDOM_DOMAIN_INITIALIZATION, 0);
    SKRandomGenerator::new(subsample_seed)
        .sample_indices_without_replacement(row_count, sample_count)
}

/// Build a [`SKTreeStructure`] over either the full view (`subsample >= 1`) or
/// the gathered dense/CSR row subset (`subsample < 1`) without densifying.
pub(crate) fn sk_build_subsampled_structure<F: SKFloat>(
    features: &SKDataView<'_, F>,
    negative_gradients: &[F],
    hessians: &[F],
    subsample: F,
    constraints: &SKStructureConstraints<F>,
    global_seed: u64,
    stage_index: usize,
    class_index: usize,
) -> Result<(SKTreeStructure<F>, Vec<F>, Vec<F>), SKError> {
    let row_count = negative_gradients.len();
    if subsample >= F::one() {
        let residual_array = Array1::from_vec(negative_gradients.to_vec());
        let structure = match features {
            SKDataView::Dense(dense) => {
                SKTreeStructure::build_structure(*dense, residual_array.view(), constraints)?
            }
            SKDataView::Sparse(sparse) => {
                SKTreeStructure::build_structure(*sparse, residual_array.view(), constraints)?
            }
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        };
        return Ok((structure, negative_gradients.to_vec(), hessians.to_vec()));
    }
    let selected =
        sk_select_stage_rows(row_count, subsample, global_seed, stage_index, class_index);
    let sampled_gradients: Vec<F> = selected
        .iter()
        .map(|&row| negative_gradients[row])
        .collect();
    let sampled_hessians: Vec<F> = selected.iter().map(|&row| hessians[row]).collect();
    let residual_array = Array1::from_vec(sampled_gradients.clone());
    let structure = match features {
        SKDataView::Dense(dense) => {
            let gathered = sk_gather_dense_rows(dense, &selected);
            SKTreeStructure::build_structure(gathered.view(), residual_array.view(), constraints)?
        }
        SKDataView::Sparse(sparse) => {
            let gathered = sk_gather_sparse_rows(sparse, &selected);
            SKTreeStructure::build_structure(gathered.view(), residual_array.view(), constraints)?
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            });
        }
    };
    Ok((structure, sampled_gradients, sampled_hessians))
}
