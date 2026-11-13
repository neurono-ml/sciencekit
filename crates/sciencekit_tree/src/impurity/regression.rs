//! Regression impurity over response vectors.

use sciencekit_common::SKFloat;

/// Squared-error impurity: the population variance of the responses.
///
/// Returns zero for an empty or constant vector.
pub fn sk_squared_error_impurity<F: SKFloat>(responses: &[F]) -> F {
    if responses.is_empty() {
        return F::zero();
    }
    let count = F::from(responses.len()).expect("response count fits in the scalar");
    let mean = responses.iter().fold(F::zero(), |sum, value| sum + *value) / count;
    let squared_deviations = responses.iter().fold(F::zero(), |sum, value| {
        let deviation = *value - mean;
        sum + deviation * deviation
    });
    squared_deviations / count
}

/// Absolute-error impurity: the mean absolute deviation from the median.
///
/// Returns zero for an empty or constant vector. The median is the middle
/// value of the sorted responses, averaging the two middle values for an even
/// count; any median of a flat region yields the same absolute deviation.
pub fn sk_absolute_error_impurity<F: SKFloat>(responses: &[F]) -> F {
    if responses.is_empty() {
        return F::zero();
    }
    let mut sorted = responses.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite responses"));
    let median = sk_median_of_sorted(&sorted);
    let count = F::from(sorted.len()).expect("response count fits in the scalar");
    let deviation = sorted
        .iter()
        .fold(F::zero(), |sum, value| sum + (*value - median).abs());
    deviation / count
}

/// The median of a sorted, non-empty slice.
pub(crate) fn sk_median_of_sorted<F: SKFloat>(sorted: &[F]) -> F {
    let length = sorted.len();
    if length % 2 == 1 {
        sorted[length / 2]
    } else {
        let middle = length / 2;
        (sorted[middle - 1] + sorted[middle]) / F::from(2).expect("two fits in the scalar")
    }
}
