//! Fit, predict, transform, score, save, load and model lifecycle entry
//! points. Carries no implementation logic.

mod fitting;
mod handles;
mod inference;
mod persistence;

pub use fitting::{sk_ffi_fit, sk_ffi_model_free};
pub use inference::{sk_ffi_predict, sk_ffi_score, sk_ffi_transform};
pub use persistence::{sk_ffi_model_load, sk_ffi_model_save};
