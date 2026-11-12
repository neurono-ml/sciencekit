//! Neumaier-compensated sparse CSR Gram and column-sum scatter accumulation
//! without densification (spec `incremental-pca`, decision 2).

use ndarray::{Array1, Array2, azip};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use super::accumulator::SKCovarianceAccumulator;
use super::merge::{sk_merge_moments, sk_symmetrized};

/// Accumulates a sparse CSR batch into `accumulator` via `Gram − n · mean · meanᵀ`
/// and the Chan-Golub-LeVeque merge, never densifying the sparse input.
pub(super) fn sk_update_sparse<F: SKFloat>(
    accumulator: &mut SKCovarianceAccumulator<F>,
    sparse: CsMatView<'_, F>,
) {
    let (rows_count, columns_count) = (sparse.rows(), sparse.cols());
    if rows_count == 0 || columns_count == 0 {
        return;
    }

    let mut column_sums = Array1::<F>::zeros(columns_count);
    let mut sum_compensation = Array1::<F>::zeros(columns_count);
    let mut gram = Array2::<F>::zeros((columns_count, columns_count));
    let mut gram_compensation = Array2::<F>::zeros((columns_count, columns_count));

    for row_view in sparse.outer_iterator() {
        for (column_left, &value_left) in row_view.iter() {
            sk_neumaier_add(
                &mut column_sums[column_left],
                &mut sum_compensation[column_left],
                value_left,
            );
            for (column_right, &value_right) in row_view.iter() {
                sk_neumaier_add(
                    &mut gram[(column_left, column_right)],
                    &mut gram_compensation[(column_left, column_right)],
                    value_left * value_right,
                );
            }
        }
    }

    azip!((sum in &mut column_sums, &compensation in &sum_compensation) {
        *sum = *sum + compensation;
    });
    azip!((entry in &mut gram, &compensation in &gram_compensation) {
        *entry = *entry + compensation;
    });

    let rows_scalar = F::from(rows_count).expect("row count fits in scalar");
    let batch_mean = column_sums.mapv(|sum| sum / rows_scalar);

    gram.rows_mut()
        .into_iter()
        .zip(batch_mean.iter())
        .for_each(|(mut gram_row, &row_mean)| {
            azip!((entry in &mut gram_row, &column_mean in &batch_mean) {
                *entry = *entry - rows_scalar * row_mean * column_mean;
            });
        });
    let batch_scatter = sk_symmetrized(gram.view());

    sk_merge_moments(
        accumulator,
        rows_count,
        batch_mean.view(),
        batch_scatter.view(),
    );
}

#[inline]
fn sk_neumaier_add<F: SKFloat>(sum: &mut F, compensation: &mut F, term: F) {
    let tentative = *sum + term;
    if sum.abs() >= term.abs() {
        *compensation = *compensation + ((*sum - tentative) + term);
    } else {
        *compensation = *compensation + ((term - tentative) + *sum);
    }
    *sum = tentative;
}
