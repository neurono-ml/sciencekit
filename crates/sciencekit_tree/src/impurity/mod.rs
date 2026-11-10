//! Impurity measures for classification and regression (spec `decision-trees`).

mod classification;
mod regression;

pub use classification::{sk_entropy_impurity, sk_gini_impurity};
pub use regression::{sk_absolute_error_impurity, sk_squared_error_impurity};

#[cfg(test)]
mod impurity_tests;
#[cfg(test)]
mod regression_impurity_tests;
