//! A Fenwick order-statistics summary for absolute-error split scans.

use sciencekit_common::SKFloat;

/// A multiset of responses supporting insertion, removal, median selection and
/// the total absolute deviation from the median in `O(log n)`.
#[derive(Debug, Clone)]
pub(crate) struct SKRankedSummary<F> {
    values: Vec<F>,
    counts: Vec<usize>,
    sums: Vec<F>,
    total_count: usize,
    total_sum: F,
}

impl<F: SKFloat> SKRankedSummary<F> {
    /// Build an empty summary over the sorted, unique response values.
    pub(crate) fn new(values: Vec<F>) -> Self {
        let size = values.len() + 1;
        SKRankedSummary {
            values,
            counts: vec![0; size],
            sums: vec![F::zero(); size],
            total_count: 0,
            total_sum: F::zero(),
        }
    }

    /// The rank of a response in the summary's value table.
    pub(crate) fn rank_of(&self, value: F) -> usize {
        self.values
            .binary_search_by(|candidate| candidate.partial_cmp(&value).expect("finite responses"))
            .expect("response belongs to the value table")
    }

    /// Insert one response of the given rank.
    pub(crate) fn insert(&mut self, rank: usize) {
        self.update(rank, 1usize, true);
    }

    /// Remove one response of the given rank.
    pub(crate) fn remove(&mut self, rank: usize) {
        self.update(rank, 1usize, false);
    }

    fn update(&mut self, rank: usize, amount: usize, add: bool) {
        let value = self.values[rank];
        let mut index = rank + 1;
        while index < self.counts.len() {
            if add {
                self.counts[index] += amount;
                self.sums[index] = self.sums[index] + value;
                self.total_count += amount;
                self.total_sum = self.total_sum + value;
            } else {
                self.counts[index] -= amount;
                self.sums[index] = self.sums[index] - value;
                self.total_count -= amount;
                self.total_sum = self.total_sum - value;
            }
            index += index & index.wrapping_neg();
        }
    }

    /// The inclusive prefix count and sum up to a rank.
    fn prefix(&self, rank: usize) -> (usize, F) {
        let mut index = rank + 1;
        let mut count = 0usize;
        let mut sum = F::zero();
        while index > 0 {
            count += self.counts[index];
            sum = sum + self.sums[index];
            index -= index & index.wrapping_neg();
        }
        (count, sum)
    }

    /// The total absolute deviation from the median.
    pub(crate) fn absolute_deviation_cost(&self) -> F {
        if self.total_count == 0 {
            return F::zero();
        }
        let lower_rank = self.select((self.total_count - 1) / 2);
        let median = self.values[lower_rank];
        let (left_count, left_sum) = self.prefix(lower_rank);
        let right_count = self.total_count - left_count;
        let right_sum = self.total_sum - left_sum;
        let left = median * F::from(left_count).expect("count fits") - left_sum;
        let right = right_sum - median * F::from(right_count).expect("count fits");
        (left + right).max(F::zero())
    }

    /// The rank of the `k`-th smallest element (zero-based).
    fn select(&self, k: usize) -> usize {
        let mut index = 0usize;
        let mut remaining = k;
        let mut step = 1usize;
        while step < self.counts.len() {
            step <<= 1;
        }
        while step > 0 {
            let next = index + step;
            if next < self.counts.len() && self.counts[next] <= remaining {
                remaining -= self.counts[next];
                index = next;
            }
            step >>= 1;
        }
        index
    }
}
