//! The zero-copy linear-operator abstraction (spec
//! `iterative-linear-solvers`).
//!
//! An operator exposes its (square) dimension and writes its product with a
//! vector into a caller-provided buffer. This lets an iterative solver drive an
//! operator without owning or copying its data.

use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use sciencekit_common::SKFloat;

/// A zero-copy linear operator over the sealed scalar.
///
/// Implementors hold their data by reference and never transfer ownership of
/// it to the solver; [`SKLinearOperator::apply_into`] writes the product of the
/// operator and `input` into `output`, both `dimension()` long.
pub trait SKLinearOperator<F: SKFloat>: Send + Sync {
    /// The operator's (square) dimension.
    fn dimension(&self) -> usize;

    /// Write `self · input` into `output`; both are [`SKLinearOperator::dimension`] long.
    fn apply_into(&self, input: ArrayView1<F>, output: ArrayViewMut1<F>);

    /// Allocating convenience form of [`SKLinearOperator::apply_into`].
    fn apply(&self, input: ArrayView1<F>) -> Array1<F> {
        let mut output = Array1::zeros(self.dimension());
        self.apply_into(input, output.view_mut());
        output
    }
}
