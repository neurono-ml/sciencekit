//! One-time representation dispatch for the linear SVC solver
//! (spec `linear-support-vector-classifier`).

use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CompressedStorage;

use super::access::SKLinearSvcDesignAccess;
use super::dense::SKDenseLinearSvcDesign;
use super::sparse::SKSparseLinearSvcDesign;

/// One-time representation dispatch for the solver.
#[derive(Debug)]
pub(crate) enum SKLinearSvcDesign<'a, F: SKFloat> {
    /// Dense design.
    Dense(SKDenseLinearSvcDesign<'a, F>),
    /// Borrowed CSR design.
    Sparse(SKSparseLinearSvcDesign<'a, F>),
}

impl<'a, F: SKFloat> SKLinearSvcDesign<'a, F> {
    /// Build the design for a dense or CSR feature view, dispatching once.
    pub(crate) fn dispatch(
        features: SKDataView<'a, F>,
        has_intercept: bool,
    ) -> Result<Self, SKError> {
        match features {
            SKDataView::Dense(dense) => Ok(Self::Dense(SKDenseLinearSvcDesign::new(
                dense,
                has_intercept,
            ))),
            SKDataView::Sparse(sparse) => {
                if sparse.storage() != CompressedStorage::CSR {
                    return Err(SKError::UnsupportedRepresentation {
                        representation: "csc",
                        suggestion: "convert the sparse matrix to CSR",
                    });
                }
                Ok(Self::Sparse(SKSparseLinearSvcDesign::new(
                    sparse,
                    has_intercept,
                )))
            }
            // `SKDataView` is `#[non_exhaustive]` in `sciencekit_common`, so the
            // wildcard is required across the crate boundary even though the
            // currently known variants are only `Dense` and `Sparse`.
            _ => Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a dense matrix or a CSR matrix",
            }),
        }
    }
}

/// Hand the solver the active representation without per-row branching.
pub(crate) fn sk_linear_svc_design_access<'design, F: SKFloat>(
    design: &'design SKLinearSvcDesign<'_, F>,
) -> &'design (dyn SKLinearSvcDesignAccess<F> + Sync) {
    match design {
        SKLinearSvcDesign::Dense(dense) => dense,
        SKLinearSvcDesign::Sparse(sparse) => sparse,
    }
}
