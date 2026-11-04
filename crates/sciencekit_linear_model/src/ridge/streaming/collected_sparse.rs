//! The in-memory sparse branch of the ridge streaming fit: accumulate the same
//! centered cross products over owned batches and solve once, without
//! densifying.

use sciencekit_common::execution::SKStreamDecision;
use sciencekit_common::{SKError, SKExecutionContext, SKSupervisedBatch};
use sciencekit_math::SKBackendSelection;

use super::batches::update_gram;
use super::state::StreamingCenteredGramState;
use crate::ridge::core_implementation::SKRidge;
use crate::ridge::model::SKRidgeModel;
use crate::ridge::solving::solve_from_gram;

impl<F: SKBackendSelection> SKRidge<F> {
    /// Accumulate every collected batch's centered cross products and solve.
    pub(super) fn fit_collected_sparse(
        &self,
        batches: Vec<SKSupervisedBatch<F>>,
    ) -> Result<SKRidgeModel<F>, SKError> {
        let machine = SKExecutionContext::real();
        let mut state =
            StreamingCenteredGramState::new(machine.available_memory_bytes, machine.cpu_cores);
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
        let solution = solve_from_gram(
            &state.gram,
            self.include_intercept,
            &self.alphas,
            self.backend,
        )?;
        Ok(SKRidgeModel::from_solve(
            self.alphas.clone(),
            solution.coefficients,
            solution.intercepts,
            solution.rank,
            solution.singular_values,
            solution.resolved,
            self.include_intercept,
            self.backend,
            self.execution_intent(),
            state.gram,
        ))
    }
}
