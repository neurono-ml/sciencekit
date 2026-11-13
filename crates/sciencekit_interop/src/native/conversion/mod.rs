//! Pure dispatcher: the native conversion surface and its validation helpers.
//! Carries no implementation logic.

mod classifiers;
mod dispatch;
mod preprocessing;
mod regressors;
mod result;
mod tensor_names;
mod validation;

pub use dispatch::{SKNativeImport, sk_import_native_state};
