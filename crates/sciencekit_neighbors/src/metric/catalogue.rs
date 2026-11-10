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

    /// The lower bound on the distance from `query` to any point inside the
    /// axis-aligned box `[minimum, maximum]`.
    ///
    /// The bound is expressed in this metric's units and is zero when the query
    /// lies inside the box.
    pub fn axis_aligned_lower_bound<F: SKFloat>(
        self,
        query: ArrayView1<F>,
        minimum: &[F],
        maximum: &[F],
    ) -> F {
        let mut total = F::zero();
        let mut largest_gap = F::zero();
        for ((&coordinate, &low), &high) in query.iter().zip(minimum).zip(maximum) {
            let gap = if coordinate < low {
                low - coordinate
            } else if coordinate > high {
                coordinate - high
            } else {
                F::zero()
            };
            match self {
                SKNeighborMetric::Euclidean | SKNeighborMetric::SquaredEuclidean => {
                    total = total + gap * gap;
                }
                SKNeighborMetric::Manhattan => total = total + gap,
                SKNeighborMetric::Chebyshev => largest_gap = largest_gap.max(gap),
            }
        }
        match self {
            SKNeighborMetric::Euclidean => total.sqrt(),
            SKNeighborMetric::Chebyshev => largest_gap,
            _ => total,
        }
    }

    /// The lower bound on the distance from `query` to any point inside the
    /// ball with the given `centroid` and Euclidean `radius`.
    ///
    /// Zero when the query lies inside the ball. The ball radius is Euclidean
    /// by construction; the metric-specific conversion keeps the bound valid
    /// for every catalogued metric (`max(0, d(q, centroid) − radius)` for the
    /// metrics whose unit is the Euclidean one, the squared and the
    /// `1 / sqrt(features)` scaling for the metric-specific cases).
    pub fn ball_lower_bound<F: SKFloat>(
        self,
        query: ArrayView1<F>,
        centroid: &[F],
        radius: F,
    ) -> F {
        let squared: F = query
            .iter()
            .zip(centroid.iter())
            .fold(F::zero(), |total, (&a, &b)| {
                let difference = a - b;
                total + difference * difference
            });
        let gap = (squared.sqrt() - radius).max(F::zero());
        match self {
            SKNeighborMetric::Euclidean | SKNeighborMetric::Manhattan => gap,
            SKNeighborMetric::SquaredEuclidean => gap * gap,
            SKNeighborMetric::Chebyshev => {
                let features_count = query.len();
                if features_count == 0 {
                    gap
                } else {
                    gap / F::from(features_count)
                        .expect("feature count fits in the scalar")
                        .sqrt()
                }
            }
        }
    }
}
