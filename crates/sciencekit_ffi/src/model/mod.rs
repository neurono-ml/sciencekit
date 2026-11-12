//! Pure dispatcher: fitted-model storage, fitting, operations and
//! persistence. Carries no implementation logic.

mod definition;
mod fitting;
mod operations;
mod persistence;

pub use definition::{SKFfiFitted, SKFfiModel, SKFfiModelKind};
pub use fitting::{dtype_name, sk_ffi_fit_dense_f32, sk_ffi_fit_dense_f64};
pub use operations::{sk_ffi_predict_fitted, sk_ffi_score_fitted, sk_ffi_transform_fitted};
pub use persistence::{sk_ffi_load_model_file, sk_ffi_save_model_file};

#[cfg(test)]
mod concurrency_tests;
#[cfg(test)]
mod model_tests;
