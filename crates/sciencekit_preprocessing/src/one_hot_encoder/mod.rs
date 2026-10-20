//! `SKOneHotEncoder`: categorical features to a numeric indicator
//! representation (spec `one-hot-encoder`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file.

mod builder;
mod categories;
mod core_implementation;
mod model;
mod output;
mod streaming;
mod unknown_policy;

pub use builder::SKOneHotEncoderBuilder;
pub use categories::{SKIntegerCategoryTable, SKOneHotColumnCategories};
pub use core_implementation::SKOneHotEncoder;
pub use model::SKOneHotEncoderModel;
pub use output::SKEncodedOutput;
pub use unknown_policy::SKUnknownPolicy;

#[cfg(test)]
mod one_hot_tests;
