//! `sciencekit_cluster` — unsupervised clustering estimators for sciencekit
//! (PRD §12, Phase 5).
//!
//! The crate opens with `SKKMeans`, the Lloyd and Elkan k-means estimator with
//! seeded k-means++ initialization and typed cluster assignments. Sparse input
//! is read through the norm-expansion distance without densification.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod kmeans;
