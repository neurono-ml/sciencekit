//! The logistic solvers and their orchestration (spec `logistic-solver`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod binary;
mod binary_saga;
mod lbfgs;
mod liblinear_dual;
mod liblinear_primal;
mod multinomial;
mod multinomial_saga;
mod orchestration;
mod saga;
mod solution;

pub(crate) use orchestration::sk_solve_logistic;
pub(crate) use solution::SKLogisticSolution;

#[cfg(test)]
mod solver_tests;
