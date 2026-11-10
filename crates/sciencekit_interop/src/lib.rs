//! `sciencekit_interop` — the engine-neutral interoperability layer (PRD §3.1,
//! spec `dataset-interop-core`).
//!
//! Heterogeneous, columnar datasets (Polars, DataFusion, Arrow, pandas, Spark)
//! arrive as named, nullable, per-column-typed tables. This crate is the single
//! seam through which any engine plugs into sciencekit's homogeneous dense,
//! sparse and streaming representations: an ordered schema with explicit
//! column roles, an ordered feature/target selection, one documented
//! missing-value policy, typed column batches, conversion into the existing
//! views and owned streaming blocks, the reopenable tabular source contract and
//! an async-to-sync bridge for async engines.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.
//!
//! No engine crate is a dependency of the core: `arrow`, `polars`,
//! `datafusion` and `memmap` are optional features added by later changes.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod batch;
pub mod bridge;
pub mod columns;
pub mod missing;
pub mod schema;
pub mod selection;
pub mod source;
pub mod views;
