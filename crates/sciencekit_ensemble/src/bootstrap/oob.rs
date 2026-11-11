//! Out-of-bag row bookkeeping for bootstrap ensembles.

/// Return the ascending training row indices absent from `drawn`.
pub(crate) fn sk_out_of_bag_rows(row_count: usize, drawn: &[usize]) -> Vec<usize> {
    let mut drawn_mask = vec![false; row_count];
    drawn.iter().for_each(|&row| {
        drawn_mask[row] = true;
    });
    (0..row_count).filter(|&row| !drawn_mask[row]).collect()
}
