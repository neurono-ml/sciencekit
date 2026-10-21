//! Pure dispatcher: the ridge solvers and their resolution. Carries no
//! implementation logic.

mod centering;
mod cholesky;
mod resolution;
mod solve_types;
mod svd;

pub(super) use cholesky::solve_with_cholesky;
pub(super) use resolution::solve_ridge;
pub(super) use solve_types::SKRidgeSolution;
