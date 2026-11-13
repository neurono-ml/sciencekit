//! Partition sizes expressed as a fraction or an explicit count.

/// The size of a train or test partition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SKPartitionSize {
    /// A fraction in the open interval `(0, 1)`.
    Fraction(f64),
    /// An explicit, strictly positive row count.
    Count(usize),
}

impl SKPartitionSize {
    /// Whether the value passes the size-independent validation.
    pub(crate) fn is_valid(&self) -> bool {
        match self {
            SKPartitionSize::Fraction(fraction) => {
                fraction.is_finite() && *fraction > 0.0 && *fraction < 1.0
            }
            SKPartitionSize::Count(count) => *count > 0,
        }
    }
}
