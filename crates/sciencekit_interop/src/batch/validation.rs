//! Up-front validation of a selected feature block (spec
//! `dataset-interop-core`).

use sciencekit_common::SKError;

use super::SKTabularBatch;
use crate::schema::SKLogicalType;

/// Reject an unsupported selected feature type once, before any element is
/// processed.
///
/// The whole selection is validated first, so a batch containing an
/// unsupported column fails on the type rather than part-way through coercion,
/// and the error names the first offending column with a conversion suggestion.
///
/// # Errors
/// Returns [`SKError::UnknownColumn`] for an out-of-range position and
/// [`SKError::UnsupportedColumnType`] for a selected nominal or category-coded
/// column.
pub fn sk_validate_numeric_feature_columns(
    batch: &SKTabularBatch,
    feature_indices: &[usize],
) -> Result<(), SKError> {
    for index in feature_indices {
        let column = batch
            .schema()
            .column_at(*index)
            .ok_or_else(|| SKError::UnknownColumn {
                column: format!("position {index}"),
                suggestion: "check the selection against the batch schema".to_string(),
            })?;
        if column.logical_type().is_numeric() {
            continue;
        }
        let suggestion = match column.logical_type() {
            SKLogicalType::Nominal => {
                "encode the nominal column as category codes in the adapter before numeric materialization"
            }
            SKLogicalType::CategoryCode => {
                "select category-coded columns through the categorical view instead of the numeric path"
            }
            _ => "select only numeric columns for the numeric feature block",
        };
        return Err(SKError::UnsupportedColumnType {
            column: column.name().to_string(),
            logical_type: column.logical_type().to_string(),
            suggestion: suggestion.to_string(),
        });
    }
    Ok(())
}
