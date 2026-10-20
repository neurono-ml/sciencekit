//! The fitted `SKPolynomialFeaturesModel` and its dense transform.

use ndarray::Array2;
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind, sk_run_operation,
};

use crate::polynomial_features::output::SKExpandedOutput;
use crate::polynomial_features::terms::build_terms;

/// Fitted polynomial-features expander: the exponent-term list, the input
/// feature count and the carried configuration.
#[derive(Debug, Clone)]
pub struct SKPolynomialFeaturesModel {
    pub(super) terms: Vec<Vec<usize>>,
    pub(super) number_of_input_features: usize,
    pub(super) sparse_output: bool,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl SKPolynomialFeaturesModel {
    /// Build a fitted model from the learned configuration. Crate-internal.
    pub(crate) fn from_configuration(
        number_of_input_features: usize,
        degree: usize,
        interaction_only: bool,
        include_bias: bool,
        sparse_output: bool,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Self {
        let terms = build_terms(
            number_of_input_features,
            degree,
            interaction_only,
            include_bias,
        );
        Self {
            terms,
            number_of_input_features,
            sparse_output,
            backend,
            execution_intent,
        }
    }

    /// The number of input features.
    pub fn number_of_input_features(&self) -> usize {
        self.number_of_input_features
    }

    /// The number of output (expanded) features.
    pub fn number_of_output_features(&self) -> usize {
        self.terms.len()
    }

    /// The exponent terms, in the documented column order.
    pub fn terms(&self) -> &[Vec<usize>] {
        &self.terms
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Whether the default transform returns a sparse matrix.
    pub fn is_sparse_output(&self) -> bool {
        self.sparse_output
    }

    /// Expand into the representation selected by the builder.
    pub fn transform<'a, F, X>(&self, features: X) -> Result<SKExpandedOutput<F>, SKError>
    where
        F: SKFloat,
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        if self.sparse_output {
            Ok(SKExpandedOutput::Sparse(
                self.transform_sparse::<F, X>(features)?,
            ))
        } else {
            Ok(SKExpandedOutput::Dense(
                self.transform_dense::<F, X>(features)?,
            ))
        }
    }

    /// Expand into a dense matrix.
    pub fn transform_dense<'a, F, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        F: SKFloat,
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        self.validate_shape(dense.ncols(), dense.nrows())?;
        sk_run_operation(
            self.operation_attributes(dense.nrows(), dense.ncols()),
            || {
                let rows_count = dense.nrows();
                let mut output = Array2::<F>::zeros((rows_count, self.terms.len()));
                for (row_index, row) in dense.outer_iter().enumerate() {
                    for (term_index, term) in self.terms.iter().enumerate() {
                        output[[row_index, term_index]] = term
                            .iter()
                            .map(|&feature| row[feature])
                            .fold(F::one(), |accumulator, value| accumulator * value);
                    }
                }
                Ok(output)
            },
        )
    }

    /// Evaluate every term for one input row, returning `(term_index, value)`
    /// pairs for the nonzero terms only.
    pub(super) fn evaluate_row<F: SKFloat>(&self, row: &[F]) -> Vec<(usize, F)> {
        self.terms
            .iter()
            .enumerate()
            .map(|(term_index, term)| {
                let value = term
                    .iter()
                    .map(|&feature| row[feature])
                    .fold(F::one(), |accumulator, value| accumulator * value);
                (term_index, value)
            })
            .filter(|(_, value)| *value != F::zero())
            .collect()
    }

    /// The observability attributes for a transform.
    pub(super) fn operation_attributes(
        &self,
        rows_count: usize,
        columns_count: usize,
    ) -> SKOperationAttributes {
        SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        }
    }

    /// Reject an input whose column count differs from the fitted model.
    pub(super) fn validate_shape(
        &self,
        columns_count: usize,
        rows_count: usize,
    ) -> Result<(), SKError> {
        if columns_count != self.number_of_input_features {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                self.number_of_input_features,
                rows_count,
                columns_count,
            ));
        }
        Ok(())
    }
}
