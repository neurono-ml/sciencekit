//! Iterative linear solvers and the zero-copy linear-operator abstraction
//! (spec `iterative-linear-solvers`, PRD §4).
//!
//! The [`SKLinearOperator`] trait lets an iterative solver drive an operator
//! without owning or copying its data; [`sk_conjugate_gradient`] solves a
//! symmetric positive-definite system through that trait, and
//! [`SKCsrNormalEquationOperator`] applies `AᵀA` over a borrowed sparse matrix
//! without ever forming or densifying it.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod conjugate_gradient;
mod linear_operator;
mod normal_equation;

pub use conjugate_gradient::sk_conjugate_gradient;
pub use linear_operator::SKLinearOperator;
pub use normal_equation::SKCsrNormalEquationOperator;

#[cfg(test)]
mod normal_equation_tests;
#[cfg(test)]
mod solvers_tests;
