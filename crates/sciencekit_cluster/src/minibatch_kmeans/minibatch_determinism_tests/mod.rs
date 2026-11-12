//! Companion tests for determinism, rayon thread-count invariance, concurrent
//! sharing, `Send + Sync` bounds and backend observability spans (spec
//! `mini-batch-k-means`, task 4.4).

mod determinism_tests;
mod fixtures;
