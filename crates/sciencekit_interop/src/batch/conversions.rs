//! Conversion of a tabular batch into the library's owned streaming blocks
//! (spec `dataset-interop-core`).

use sciencekit_common::{SKDataBatch, SKError, SKFeatureBlock, SKFloat, SKSupervisedBatch};

use super::{SKFeatureLayout, SKTabularBatch};
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;

impl SKTabularBatch {
    /// Convert the selected features into an owned streaming data block.
    ///
    /// The block fully owns its buffers and carries the batch's position and
    /// final indication, so it stays valid after the source advances.
    ///
    /// # Errors
    /// Propagates selection, type, policy and materialization failures.
    pub fn to_data_batch<F: SKFloat>(
        &self,
        selection: &SKColumnSelection,
        policy: SKMissingValuePolicy,
        layout: SKFeatureLayout,
    ) -> Result<SKDataBatch<F>, SKError> {
        let block = self.to_feature_block(selection, policy, layout)?;
        let data_batch = match block {
            SKFeatureBlock::Dense(matrix) => {
                SKDataBatch::new_dense(matrix, self.position(), self.is_final())
            }
            SKFeatureBlock::Sparse(matrix) => {
                SKDataBatch::new_sparse(matrix, self.position(), self.is_final())
            }
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the block to dense or CSR before streaming",
                });
            }
        };
        Ok(data_batch)
    }

    /// Convert the selected features and target into an owned supervised block.
    ///
    /// Feature order follows the selection order; the target is coerced or
    /// canonicalized under the declared policy.
    ///
    /// # Errors
    /// Returns [`SKError::UnsupportedOperation`] when the selection has no
    /// target, and propagates conversion failures.
    pub fn to_supervised_batch<F: SKFloat>(
        &self,
        selection: &SKColumnSelection,
        policy: SKMissingValuePolicy,
        layout: SKFeatureLayout,
    ) -> Result<SKSupervisedBatch<F>, SKError> {
        if !selection.has_target() {
            return Err(SKError::UnsupportedOperation {
                operation: "supervised_conversion",
                reason: "the selection does not declare a target column".to_string(),
            });
        }
        let targets = self
            .extract_target::<F>(selection, policy)?
            .to_scalar_values()?;
        let block = self.to_feature_block(selection, policy, layout)?;
        match block {
            SKFeatureBlock::Dense(matrix) => {
                SKSupervisedBatch::new_dense(matrix, targets, self.position(), self.is_final())
            }
            SKFeatureBlock::Sparse(matrix) => {
                SKSupervisedBatch::new_sparse(matrix, targets, self.position(), self.is_final())
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the block to dense or CSR before streaming",
            }),
        }
    }
}
