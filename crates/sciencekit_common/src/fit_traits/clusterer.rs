//! Cluster assignment contract (spec `clustering-contracts`).
//!
//! A fitted clusterer exposes one [`SKClusterLabel`] per input row. The label is
//! a typed value: a cluster identifier or the explicit [`SKClusterLabel::NOISE`]
//! variant — never a numeric sentinel. The identifier stores `index + 1` in a
//! `NonZeroU32`, so the enum's niche carries the noise variant and the whole
//! label stays four bytes wide.

use ndarray::{Array1, ArrayView1};
use std::num::NonZeroU32;

use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;

/// A validated cluster identifier.
///
/// Stores the public zero-based index shifted by one inside a `NonZeroU32`, so
/// cluster `0` is representable while the zero niche stays free for
/// [`SKClusterLabel::NOISE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SKClusterIdentifier(NonZeroU32);

impl SKClusterIdentifier {
    /// The largest cluster index a label can carry.
    pub const MAXIMUM_INDEX: u32 = u32::MAX - 1;

    /// Builds an identifier from a zero-based cluster index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds [`Self::MAXIMUM_INDEX`]. Callers converting
    /// external data use [`SKClusterLabel::from_i64`], which reports a
    /// structured error instead.
    pub fn new(index: u32) -> Self {
        assert!(
            index <= Self::MAXIMUM_INDEX,
            "cluster index {index} exceeds the maximum {}",
            Self::MAXIMUM_INDEX
        );
        Self(NonZeroU32::new(index + 1).expect("index + 1 never wraps"))
    }

    /// Returns the zero-based cluster index.
    pub fn index(self) -> u32 {
        self.0.get() - 1
    }
}

/// One cluster assignment: a cluster identifier or explicit noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKClusterLabel {
    /// The row belongs to a cluster.
    Cluster(SKClusterIdentifier),
    /// The row was classified as noise (for example by DBSCAN).
    Noise,
}

impl SKClusterLabel {
    /// The explicit noise assignment.
    pub const NOISE: Self = Self::Noise;

    /// Builds a cluster assignment from a zero-based cluster index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds [`SKClusterIdentifier::MAXIMUM_INDEX`].
    pub fn cluster(index: u32) -> Self {
        Self::Cluster(SKClusterIdentifier::new(index))
    }

    /// Reports whether this assignment is the explicit noise variant.
    pub fn is_noise(self) -> bool {
        matches!(self, Self::Noise)
    }

    /// Returns the cluster index, or `None` for noise.
    pub fn cluster_index(self) -> Option<u32> {
        match self {
            Self::Cluster(identifier) => Some(identifier.index()),
            Self::Noise => None,
        }
    }

    /// Converts to the external `i64` convention (`Noise ↔ -1`).
    pub fn to_i64(self) -> i64 {
        match self {
            Self::Cluster(identifier) => i64::from(identifier.index()),
            Self::Noise => -1,
        }
    }

    /// Converts from the external `i64` convention.
    ///
    /// `-1` becomes [`Self::NOISE`]; a non-negative value becomes a cluster.
    /// A value below `-1` returns a structured [`SKError::Conversion`] instead
    /// of inventing a cluster.
    pub fn from_i64(value: i64) -> Result<Self, SKError> {
        if value == -1 {
            return Ok(Self::Noise);
        }
        if value < -1 {
            return Err(SKError::Conversion(format!(
                "cluster assignment {value} is below the -1 noise convention"
            )));
        }
        if value > i64::from(SKClusterIdentifier::MAXIMUM_INDEX) {
            return Err(SKError::Conversion(format!(
                "cluster assignment {value} exceeds the maximum index {}",
                SKClusterIdentifier::MAXIMUM_INDEX
            )));
        }
        let index = u32::try_from(value).map_err(|_| {
            SKError::Conversion(format!(
                "cluster assignment {value} does not fit a 32-bit cluster index"
            ))
        })?;
        Ok(Self::cluster(index))
    }
}

/// Cluster assignment lives on the fitted model type.
///
/// The method accepts the same fallible [`SKDataView`] conversion seam as the
/// other predictors, so dense and CSR inputs share one entry point.
pub trait SKClustererPredictor<F: SKFloat> {
    /// The inference error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Assigns one cluster label per input row.
    fn predict_clusters<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}

/// Validates that an assignment vector has one label per feature row.
///
/// Scorers share this helper so the shape rule — and its error — is defined in
/// exactly one place.
pub fn sk_validate_cluster_assignments(
    number_of_rows: usize,
    assignments: ArrayView1<'_, SKClusterLabel>,
) -> Result<(), SKError> {
    if assignments.len() == number_of_rows {
        return Ok(());
    }
    Err(SKError::ShapeMismatch {
        expected: vec![number_of_rows],
        found: vec![assignments.len()],
    })
}
