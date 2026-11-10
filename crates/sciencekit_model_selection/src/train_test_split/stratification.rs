//! Proportional allocation of class members to a partition.

/// Allocates `target_total` rows across classes of the given sizes.
///
/// The allocation uses the largest-remainder method: every class receives the
/// integer part of its exact share and the leftover rows go to the classes with
/// the largest remainders, ties broken by class order. The result never exceeds
/// a class's capacity and sums exactly to `target_total`.
pub(crate) fn sk_allocate_class_counts(
    class_sizes: &[usize],
    target_total: usize,
    total_rows: usize,
) -> Vec<usize> {
    if total_rows == 0 || target_total == 0 {
        return vec![0; class_sizes.len()];
    }

    let dividends: Vec<usize> = class_sizes.iter().map(|size| size * target_total).collect();
    let mut counts: Vec<usize> = dividends.iter().map(|value| value / total_rows).collect();
    let mut order: Vec<usize> = (0..class_sizes.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(dividends[index] % total_rows));

    let mut leftover = target_total - counts.iter().sum::<usize>();
    let mut position = 0;
    while leftover > 0 {
        let index = order[position % order.len()];
        if counts[index] < class_sizes[index] {
            counts[index] += 1;
            leftover -= 1;
        }
        position += 1;
    }
    counts
}
