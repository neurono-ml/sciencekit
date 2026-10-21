//! `sciencekit_linear_model` — linear models (PRD §12 Phase 2).
//!
//! The crate opens Wave 2 with [`linear_regression::SKLinearRegression`], the
//! library's first supervised estimator: a builder-configured, backend-carrying
//! ordinary-least-squares regressor with a distinct fitted model, reachable
//! identically from a dense `fit`, an incremental `partial_fit` and an
//! out-of-core `fit_streaming`.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![warn(missing_docs)]

pub mod linear_regression;
