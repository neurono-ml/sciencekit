//! The `SKMissingIndicator` estimator (spec `missing-indicator`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod configuration;
mod fitting;

pub use configuration::SKMissingIndicator;
