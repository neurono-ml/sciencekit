//! The closed supported-metric catalogue and its distance geometry.

use ndarray::ArrayView1;
use sciencekit_common::SKFloat;

/// The supported distance metrics.
///
/// The catalogue is closed because tree pruning is only sound for distances
/// obeying the triangle inequality: cosine distance and NaN-aware Euclidean
/// distance are deliberately absent. A caller who requests them explicitly is
/// rejected at build time through [`SKRequestedNeighborMetric`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKNeighborMetric {
    /// The Euclidean (`L2`) distance.
    Euclidean,
    /// The squared Euclidean distance (the Euclidean distance without the
    /// square root).
    SquaredEuclidean,
    /// The Manhattan (`L1`) distance.
    Manhattan,
    /// The Chebyshev (`L∞`) distance.
    Chebyshev,
}

impl SKNeighborMetric {
    /// Every supported metric, in catalogue order.
    pub const ALL: [SKNeighborMetric; 4] = [
        SKNeighborMetric::Euclidean,
        SKNeighborMetric::SquaredEuclidean,
        SKNeighborMetric::Manhattan,
        SKNeighborMetric::Chebyshev,
    ];

    /// The distance between two points in this metric.
    pub fn point_distance<F: SKFloat>(self, left: ArrayView1<F>, right: ArrayView1<F>) -> F {
        match self {
            SKNeighborMetric::Euclidean => {
                let squared = self.squared_euclidean(left, right);
                squared.sqrt()
            }
            SKNeighborMetric::SquaredEuclidean => self.squared_euclidean(left, right),
            SKNeighborMetric::Manhattan => left
                .iter()
                .zip(right.iter())
                .fold(F::zero(), |total, (&a, &b)| total + (a - b).abs()),
            SKNeighborMetric::Chebyshev => left
                .iter()
                .zip(right.iter())
                .fold(F::zero(), |maximum, (&a, &b)| maximum.max((a - b).abs())),
        }
    }

    fn squared_euclidean<F: SKFloat>(self, left: ArrayView1<F>, right: ArrayView1<F>) -> F {
        left.iter()
            .zip(right.iter())
            .fold(F::zero(), |total, (&a, &b)| {
                let difference = a - b;
                total + difference * difference
            })
    }

}
