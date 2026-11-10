//! The internal contract a scoring policy implements for one fold.

use sciencekit_common::{SKDataView, SKError, SKFloat, SKTargetView};

/// Scores a freshly fitted fold model against the fold's test targets.
///
/// Both public scoring enums implement this contract, so candidate/fold
/// evaluation is written once and shared by the classifier and regressor grid
/// searches.
pub(crate) trait SKFoldScorer<F: SKFloat, M> {
    /// Scores `model` on `features` against `targets`.
    fn sk_score(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<F, SKError>;
}
