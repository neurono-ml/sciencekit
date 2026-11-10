//! Acceptance suite for the sparse pairwise distances (PRD §8.7).
//!
//! Covers tiny inputs with empty rows and a single stored entry, large inputs
//! (at least 10 000 rows), every metric, the four representation combinations
//! and concurrent execution. Each case compares against the dense kernel on the
//! same logical matrix within the documented tolerance.
//!
//! This is a **pure dispatcher** module: it only declares the acceptance
//! submodules.

mod concurrency;
mod large_data;
mod tiny_data;
