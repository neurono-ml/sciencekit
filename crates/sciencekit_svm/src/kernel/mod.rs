//! Kernel support-vector helpers and owned feature storage
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod distance;
mod resolved;
mod validation;

pub use distance::{sk_evaluate_kernel_diagonal, sk_evaluate_kernel_row};
pub use resolved::SKOwnedFeatureBlock;
pub use validation::{
    sk_resolve_svm_execution_plan, sk_svm_view_dimensions, sk_validate_svm_finiteness,
    sk_validate_svm_query_features, sk_validate_svm_training_features,
};

#[cfg(test)]
mod kernel_tests;
