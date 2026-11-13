//! Raw-score evaluation and shrunk stage updates over dense and CSR views.

use sciencekit_common::{SKDataView, SKFloat};
use sciencekit_tree::SKTree;
use sprs::CsVecView;

/// Add `learning_rate * tree(row)` into `raw_scores` for every row in `features`
/// without densifying sparse inputs.
pub(crate) fn sk_update_raw_scores<F: SKFloat>(
    features: &SKDataView<'_, F>,
    tree: &SKTree<F>,
    learning_rate: F,
    raw_scores: &mut [F],
) {
    match features {
        SKDataView::Dense(dense) => {
            raw_scores
                .iter_mut()
                .enumerate()
                .for_each(|(row, raw_score)| {
                    let step = tree.leaf_value_for(|feature| dense[[row, feature]]);
                    *raw_score = *raw_score + learning_rate * step;
                });
        }
        SKDataView::Sparse(sparse) => {
            raw_scores
                .iter_mut()
                .enumerate()
                .for_each(|(row, raw_score)| {
                    let sparse_row = sparse.outer_view(row).expect("row within bounds");
                    let step = tree.leaf_value_for(|feature| sk_sparse_entry(&sparse_row, feature));
                    *raw_score = *raw_score + learning_rate * step;
                });
        }
        _ => {}
    }
}

/// Read one feature from a borrowed CSR row, treating absent coordinates as zero.
pub(crate) fn sk_sparse_entry<F: SKFloat>(row: &CsVecView<'_, F>, feature: usize) -> F {
    match row.indices().binary_search(&feature) {
        Ok(position) => row.data()[position],
        Err(_) => F::zero(),
    }
}
