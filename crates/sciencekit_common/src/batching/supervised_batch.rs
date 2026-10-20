//! The owned supervised streaming block (spec `streaming-batches`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// An owned supervised streaming block: a feature matrix paired with its
/// targets.
///
/// Like [`SKDataBatch`](super::SKDataBatch) it fully owns its data (no borrowing
/// from the source) so the pair can be moved to another thread while the source
/// advances; it carries its position in the sequence and a final-block
/// indication. Supervised algorithms that stream out-of-core (the first being
/// linear regression) consume it.
#[derive(Debug, Clone)]
pub struct SKSupervisedBatch<F> {
    features: Array2<F>,
    targets: Array1<F>,
    position: usize,
    is_final: bool,
}

impl<F> SKSupervisedBatch<F> {
    /// Create a supervised batch.
    pub fn new(features: Array2<F>, targets: Array1<F>, position: usize, is_final: bool) -> Self {
        SKSupervisedBatch {
            features,
            targets,
            position,
            is_final,
        }
    }

    /// Borrow the batch's feature matrix.
    pub fn features(&self) -> ArrayView2<F> {
        self.features.view()
    }

    /// Borrow the batch's targets.
    pub fn targets(&self) -> ArrayView1<F> {
        self.targets.view()
    }

    /// The zero-based position of this batch in the sequence.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether this is the final block of a finite source.
    pub fn is_final(&self) -> bool {
        self.is_final
    }
}
