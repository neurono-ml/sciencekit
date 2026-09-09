//! Data boundary: the canonical representation of features (spec
//! `data-view-boundary`).
//!
//! [`SKDataView`] is a `#[non_exhaustive]` enum covering dense and sparse
//! inputs. Public operations accept any type that converts into it via the
//! standard fallible bound (`TryInto`): native types convert infallibly and
//! are promoted automatically by the std blanket; third parties implement
//! [`TryFrom`] for their own types, including fallible conversions whose error
//! flows into the operation's `Result`.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod conversions;
// Private implementation module shadowing the parent name; allowed because the
// module stays private and every public item is re-exported above, so the
// same-name nesting never surfaces in public paths.
#[allow(clippy::module_inception)]
mod data_view;

pub use data_view::SKDataView;

#[cfg(test)]
mod data_view_tests;
