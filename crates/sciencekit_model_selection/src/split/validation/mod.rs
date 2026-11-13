//! Shared split validation and fold-partition helpers.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod inputs;
mod partitioning;

pub(crate) use inputs::{sk_class_groups, sk_validate_finite_features, sk_validate_finite_targets};
pub(crate) use partitioning::{
    sk_number_of_rows, sk_partition_into_chunks, sk_split_from_test, sk_validate_cardinality,
    sk_validate_non_empty, sk_validate_number_of_splits,
};
