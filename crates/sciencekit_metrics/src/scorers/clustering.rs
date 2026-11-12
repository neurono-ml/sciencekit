//! Unsupervised scorer adapters over typed cluster assignments.

use ndarray::ArrayView1;
use sciencekit_common::{
    SKClusterLabel, SKClustererPredictor, SKDataView, SKError, SKFloat, SKUnsupervisedScorer,
};

use crate::clustering::sk_silhouette_score;

/// The silhouette score over a clusterer's typed assignments.
///
/// **Noise policy:** delegates to [`sk_silhouette_score`], which excludes noise
/// rows; noise is never treated as a cluster. Fewer than two non-noise clusters are
/// rejected with [`SKError::InvalidHyperparameter`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SKSilhouetteScorer;

impl<F: SKFloat, M: SKClustererPredictor<F>> SKUnsupervisedScorer<F, M> for SKSilhouetteScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_assignments(
        &self,
        features: SKDataView<'_, F>,
        assignments: ArrayView1<'_, SKClusterLabel>,
    ) -> Result<f64, Self::Error> {
        let score = sk_silhouette_score(features, assignments)?;
        score
            .to_f64()
            .ok_or_else(|| SKError::Conversion("silhouette score does not fit in f64".to_string()))
    }
}
