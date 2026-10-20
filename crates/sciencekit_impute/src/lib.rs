//! Missing-data estimators (PRD §3.1 `sciencekit_impute`, wave `simple-imputers`).
//!
//! Two transformers: [`simple_imputer::SKSimpleImputer`], which fills the missing
//! cells of a numeric feature matrix with a per-column strategy statistic, and
//! [`missing_indicator::SKMissingIndicator`], which reports where those cells are.
//! Both are builder-only, generic over the scalar `SKFloat`, sparse-aware through
//! [`sprs`] and record the configured backend kind in every operation span.
