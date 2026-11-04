//! The explicit warm-start source of the SGD classifier (spec decision 13).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKError, SKFloat, SKLabelTable};

use super::targets::sk_number_of_heads;

/// Coefficients and intercepts shaped for a class table, used to seed a fit.
///
/// The estimator stays immutable: a warm start is an explicit value supplied to
/// the builder, never mutable instance state. A source whose class table or
/// feature count disagrees with the fit is rejected by `fit`.
#[derive(Debug, Clone)]
pub struct SKSGDClassifierCoefficients<F> {
    pub(crate) classes: SKLabelTable,
    pub(crate) coefficients: Array2<F>,
    pub(crate) intercepts: Array1<F>,
}

impl<F: SKFloat> SKSGDClassifierCoefficients<F> {
    /// Assemble a warm-start source from a class table and its heads.
    ///
    /// The number of heads must match the class table (one for a binary fit,
    /// one per class otherwise) and the intercept count must match the heads.
    pub fn new(
        classes: SKLabelTable,
        coefficients: ArrayView2<'_, F>,
        intercepts: ArrayView1<'_, F>,
    ) -> Result<Self, SKError> {
        let expected_heads = sk_number_of_heads(classes.number_of_classes());
        if coefficients.nrows() != expected_heads || intercepts.len() != coefficients.nrows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![expected_heads, coefficients.ncols()],
                found: vec![coefficients.nrows(), intercepts.len()],
            });
        }
        Ok(Self {
            classes,
            coefficients: coefficients.to_owned(),
            intercepts: intercepts.to_owned(),
        })
    }

    /// The class table the coefficients were produced for.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// The coefficients, shaped `heads × features`.
    pub fn coefficients(&self) -> ArrayView2<'_, F> {
        self.coefficients.view()
    }

    /// The intercepts, one per head.
    pub fn intercepts(&self) -> ArrayView1<'_, F> {
        self.intercepts.view()
    }

    /// The number of features the coefficients were fitted on.
    pub fn feature_count(&self) -> usize {
        self.coefficients.ncols()
    }
}
