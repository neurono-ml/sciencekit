//! The regressor's criterion implementation.

use sciencekit_common::SKFloat;

use crate::criterion::SKTreeCriterion as SKConfiguredCriterion;
use crate::growth::SKTreeCriterion;
use crate::impurity::{sk_absolute_error_impurity, sk_median_of_sorted, sk_squared_error_impurity};
use crate::splitter::SKScanContext;

/// Regression statistics backing growth and leaf outputs.
pub(crate) struct SKRegressorCriterion<'a, F> {
    pub(crate) responses: &'a [F],
    pub(crate) criterion: SKConfiguredCriterion,
}

impl<F: SKFloat> SKTreeCriterion<F> for SKRegressorCriterion<'_, F> {
    fn impurity(&self, sample_indices: &[usize]) -> F {
        match self.criterion {
            SKConfiguredCriterion::AbsoluteError => {
                sk_absolute_error_impurity(&sk_responses(self.responses, sample_indices))
            }
            _ => sk_squared_error_impurity(&sk_responses(self.responses, sample_indices)),
        }
    }

    fn leaf_output(&self, sample_indices: &[usize]) -> Vec<F> {
        match self.criterion {
            SKConfiguredCriterion::AbsoluteError => {
                let mut sorted = sk_responses(self.responses, sample_indices);
                sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite responses"));
                vec![sk_median_of_sorted(&sorted)]
            }
            _ => {
                let values = sk_responses(self.responses, sample_indices);
                let total = values.iter().fold(F::zero(), |sum, value| sum + *value);
                vec![total / F::from(values.len()).expect("count fits")]
            }
        }
    }

    fn scan_context(&self) -> SKScanContext<'_, F> {
        SKScanContext::Regression {
            responses: self.responses,
            criterion: self.criterion,
        }
    }
}

/// The responses of a node's samples.
fn sk_responses<F: SKFloat>(responses: &[F], sample_indices: &[usize]) -> Vec<F> {
    sample_indices.iter().map(|&row| responses[row]).collect()
}
