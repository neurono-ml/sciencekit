//! Which feature columns the mask covers (spec `missing-indicator`).

/// The feature scope of a fitted missingness mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKMissingFeatureScope {
    /// Mask only the features that carried at least one missing cell at fit
    /// time; this is the default.
    #[default]
    MissingOnly,
    /// Mask every feature, missing or not.
    All,
}
