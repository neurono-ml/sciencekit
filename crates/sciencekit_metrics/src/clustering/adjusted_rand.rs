//! The adjusted Rand index (spec `clustering-metrics`).

use ndarray::ArrayView1;
use sciencekit_common::SKError;

use super::contingency::{sk_contingency_table, sk_pair_count};

/// The adjusted Rand index between two labelings.
///
/// The index counts agreeing pairs and corrects for chance, so it is `1` for
/// identical labelings (up to permutation), near `0` for independent ones and
/// can be negative for worse-than-chance agreement. Both trivial labelings
/// (every row in one cluster) score `1`.
///
/// **Noise policy:** a label is just a label here; the caller decides whether
/// noise (`-1`) should be filtered before calling.
pub fn sk_adjusted_rand_score(
    labels_true: ArrayView1<'_, i64>,
    labels_pred: ArrayView1<'_, i64>,
) -> Result<f64, SKError> {
    let table = sk_contingency_table(labels_true, labels_pred, "adjusted_rand_index")?;
    if table.number_of_rows < 2 {
        return Ok(1.0);
    }

    let observed: f64 = table
        .cells
        .iter()
        .map(|(_, _, count)| sk_pair_count(*count))
        .sum();
    let true_pairs: f64 = table
        .true_counts
        .iter()
        .map(|count| sk_pair_count(*count))
        .sum();
    let predicted_pairs: f64 = table
        .predicted_counts
        .iter()
        .map(|count| sk_pair_count(*count))
        .sum();

    let total_pairs = sk_pair_count(table.number_of_rows);
    let expected = true_pairs * predicted_pairs / total_pairs;
    let denominator = 0.5 * (true_pairs + predicted_pairs) - expected;
    if denominator == 0.0 {
        return Ok(1.0);
    }
    Ok((observed - expected) / denominator)
}
