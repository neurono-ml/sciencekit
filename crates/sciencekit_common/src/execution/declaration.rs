//! The access-pattern declaration contract (spec `access-pattern`).
//!
//! An estimator implements [`SKDeclaredAccessPattern`] once; every operation reads the
//! declaration instead of hardcoding a pattern, and the conformance test compares it with
//! the parity matrix tier.

use super::modes::SKAccessPattern;

/// Declares the data-access pattern an estimator is designed for.
pub trait SKDeclaredAccessPattern {
    /// The pattern this estimator declares, identical for every operation.
    fn access_pattern(&self) -> SKAccessPattern;

    /// The documented approximate alternative for a `FullMemory` estimator.
    ///
    /// Estimators whose mathematics needs the whole design resident name the
    /// out-of-core approximation callers can use instead; every other pattern
    /// returns `None`.
    fn approximate_alternative(&self) -> Option<&'static str> {
        None
    }
}
