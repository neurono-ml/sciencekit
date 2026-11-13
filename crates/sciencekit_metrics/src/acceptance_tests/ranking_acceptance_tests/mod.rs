//! Companion acceptance tests for the ranking metrics (PRD §8.7 / §10.3),
//! split into a folder module to respect the 200-line limit.
//!
//! Covers large data, f32/f64 parity, concurrent calls and thread safety.

mod concurrency;
mod fixtures;
mod large_data;
mod precision_parity;
