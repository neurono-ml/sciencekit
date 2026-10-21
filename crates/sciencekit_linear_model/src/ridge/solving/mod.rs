//! Pure dispatcher: the ridge solvers and their resolution. Carries no
//! implementation logic.

mod centering;
mod cholesky;
mod resolution;
mod solve_types;
mod svd;

pub(super) use resolution::{reject_sparse, solve_ridge};
