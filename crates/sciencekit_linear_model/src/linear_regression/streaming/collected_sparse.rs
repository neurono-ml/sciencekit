//! The in-memory sparse branch of the OLS streaming fit: accumulate the same
//! Gram state over owned batches and solve once, without densifying.

use sciencekit_common::execution::SKStreamDecision;
use sciencekit_common::{SKError, SKExecutionContext, SKFloat, SKSupervisedBatch};
use sciencekit_math::SKBackendSelection;

use super::batches::update_gram;
use super::state::StreamingGramState;
use crate::linear_regression::core_implementation::SKLinearRegression;
use crate::linear_regression::model::SKLinearRegressionModel;

impl SKLinearRegression {
    /// Accumulate every collected batch's cross products and solve once.
    pub(super) fn fit_collected_sparse<F>(
        &self,
        batches: Vec<SKSupervisedBatch<F>>,
    ) -> Result<SKLinearRegressionModel<F>, SKError>
    where
        F: SKFloat + SKBackendSelection,
    {
        let machine = SKExecutionContext::real();
        let mut state = StreamingGramState::new(machine.available_memory_bytes, machine.cpu_cores);
        for batch in batches {
            if update_gram(self, batch, &mut state) == SKStreamDecision::Stop {
                break;
            }
        }
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.gram.row_count == 0 {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        SKLinearRegressionModel::from_gram(
            state.gram,
            self.include_intercept,
            self.backend,
            self.execution_intent(),
        )
    }
}
