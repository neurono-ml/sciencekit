//! Cross-cutting acceptance tests for KMeans (spec `k-means`, PRD §8.7).

mod concurrency_tests;
mod determinism_tests;
mod fixtures;
mod large_data_tests;
mod metric_tests;
mod representation_tests;
mod tiny_data_tests;
