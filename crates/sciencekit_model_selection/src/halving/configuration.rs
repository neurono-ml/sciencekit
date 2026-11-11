//! Configuration shared by the classifier and regressor halving searches.

use ndarray::Array1;
use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat};

use crate::SKSplitter;
use crate::grid_search::{SKClassifierScoring, SKErrorScore, SKRegressorScoring, SKScoringFamily};

use super::SKHalvingResource;

/// Scoring policy wrapper accepted by the halving grid search builders so
/// family mismatches are validated cleanly at construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKHalvingScoring {
    /// Classification scoring policy.
    Classifier(SKClassifierScoring),
    /// Regression scoring policy.
    Regressor(SKRegressorScoring),
}

impl SKHalvingScoring {
    /// The estimator family this scoring policy belongs to.
    pub fn family(self) -> SKScoringFamily {
        match self {
            SKHalvingScoring::Classifier(policy) => policy.family(),
            SKHalvingScoring::Regressor(policy) => policy.family(),
        }
    }

    pub(crate) fn into_classifier(self) -> Result<SKClassifierScoring, SKError> {
        match self {
            SKHalvingScoring::Classifier(policy) => Ok(policy),
            SKHalvingScoring::Regressor(_) => Err(SKError::InvalidHyperparameter {
                name: "scoring",
                reason: "a regressor scoring policy cannot drive a classifier halving search"
                    .to_owned(),
            }),
        }
    }

    pub(crate) fn into_regressor(self) -> Result<SKRegressorScoring, SKError> {
        match self {
            SKHalvingScoring::Regressor(policy) => Ok(policy),
            SKHalvingScoring::Classifier(_) => Err(SKError::InvalidHyperparameter {
                name: "scoring",
                reason: "a classifier scoring policy cannot drive a regressor halving search"
                    .to_owned(),
            }),
        }
    }
}

impl From<SKClassifierScoring> for SKHalvingScoring {
    fn from(policy: SKClassifierScoring) -> Self {
        SKHalvingScoring::Classifier(policy)
    }
}

impl From<SKRegressorScoring> for SKHalvingScoring {
    fn from(policy: SKRegressorScoring) -> Self {
        SKHalvingScoring::Regressor(policy)
    }
}

/// Family-independent configuration frozen by the halving search builders.
#[allow(dead_code)]
pub(crate) struct SKHalvingGridSearchConfiguration<F: SKFloat, E> {
    pub(crate) candidates: Vec<E>,
    pub(crate) splitter: Box<dyn SKSplitter<F> + Send + Sync>,
    pub(crate) factor: usize,
    pub(crate) minimum_resource: usize,
    pub(crate) maximum_resource: Option<usize>,
    pub(crate) resource: SKHalvingResource,
    pub(crate) aggressive_elimination: bool,
    pub(crate) random_seed: u64,
    pub(crate) refit: bool,
    pub(crate) error_score: SKErrorScore<F>,
    pub(crate) groups: Option<Array1<i64>>,
    pub(crate) execution_mode: SKExecutionMode,
    pub(crate) backend: SKBackendKind,
}

impl<F: SKFloat, E> SKHalvingGridSearchConfiguration<F, E> {
    pub(crate) fn number_of_candidates(&self) -> usize {
        self.candidates.len()
    }
}
