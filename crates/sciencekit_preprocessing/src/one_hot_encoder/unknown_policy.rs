//! Unknown-category handling policy for the one-hot encoder.

/// How the encoder treats a category not present in a column's fitted table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKUnknownPolicy {
    /// Fail with a structured error (the default).
    #[default]
    Reject,
    /// Encode the entry as an all-zero block for that column.
    Ignore,
}

impl SKUnknownPolicy {
    /// Whether the policy fails on an unknown category.
    pub fn rejects(&self) -> bool {
        matches!(self, SKUnknownPolicy::Reject)
    }
}
