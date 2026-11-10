//! Configuration shared by the classifier and regressor grid searches.

use ndarray::Array1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::SKErrorScore;
use crate::SKSplitter;

/// The family-independent grid-search configuration.
///
/// Candidates are fully configured estimators; the splitter is boxed so both
/// wrappers share one configuration type without exposing a generic splitter
/// parameter in the public estimator surface.
pub(crate) struct SKGridSearchConfiguration<F: SKFloat, E> {
    pub(crate) candidates: Vec<E>,
    pub(crate) splitter: Box<dyn SKSplitter<F>>,
    pub(crate) refit: bool,
    pub(crate) error_score: SKErrorScore<F>,
    pub(crate) groups: Option<Array1<i64>>,
    pub(crate) execution_mode: SKExecutionMode,
    pub(crate) backend: SKBackendKind,
}

impl<F: SKFloat, E> SKGridSearchConfiguration<F, E> {
    pub(crate) fn number_of_candidates(&self) -> usize {
        self.candidates.len()
    }
}
