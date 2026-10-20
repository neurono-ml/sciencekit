//! The univariate imputation strategies (spec `simple-imputer`).

/// The per-column statistic that fills the missing cells.
///
/// Mirrors `sklearn.impute.SimpleImputer`'s `strategy` set: `Mean` (default),
/// `Median`, `MostFrequent` (ties break toward the smallest value) and
/// `Constant` (the `fill_value`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKImputationStrategy {
    /// The mean of the observed values of the column.
    Mean,
    /// The median (order statistic of the sorted observed values).
    Median,
    /// The most frequent observed value; ties break toward the smallest.
    MostFrequent,
    /// The configured constant `fill_value` (default `0.0`).
    Constant,
}
