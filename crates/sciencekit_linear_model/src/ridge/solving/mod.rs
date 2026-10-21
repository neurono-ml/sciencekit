//! Pure dispatcher: the ridge solvers and their resolution. Carries no
//! implementation logic.

mod centering;
mod cholesky;
mod gram_solve;
mod resolution;
mod solve_types;
mod svd;

pub(crate) use gram_solve::solve_from_gram;
pub(super) use resolution::solve_ridge;
pub(super) use solve_types::SKRidgeSolution;
