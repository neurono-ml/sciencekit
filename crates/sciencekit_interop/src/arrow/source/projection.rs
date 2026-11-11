//! Column projection for Arrow sources (spec `arrow-columnar-interop`).
//!
//! A source resolves the caller's selection once per pass against the Arrow
//! schema and validates the logical types of only the selected columns, so
//! unselected columns carrying unsupported Arrow types never block a projected
//! read.

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema};
use sciencekit_common::SKError;

use super::super::data_type_mapping::sk_logical_type_from_arrow_data_type;
use crate::schema::{SKColumnRole, SKColumnSchema, SKTabularSchema};
use crate::selection::{SKColumnReference, SKColumnSelection};

fn sk_resolve_reference_in_arrow_schema(
    schema: &Schema,
    reference: &SKColumnReference,
) -> Result<usize, SKError> {
    let fields = schema.fields();
    match reference {
        SKColumnReference::Name(name) => {
            let mut matches = fields
                .iter()
                .enumerate()
                .filter(|(_, field)| field.name() == name)
                .map(|(index, _)| index);
            let Some(first) = matches.next() else {
                return Err(SKError::UnknownColumn {
                    column: name.clone(),
                    suggestion: "check the schema for the exact column name".to_string(),
                });
            };
            if matches.next().is_some() {
                return Err(SKError::DuplicateColumn {
                    column: name.clone(),
                    suggestion: "rename one of the duplicate columns at the source".to_string(),
                });
            }
            Ok(first)
        }
        SKColumnReference::Position(position) => {
            if *position < fields.len() {
                Ok(*position)
            } else {
                Err(SKError::UnknownColumn {
                    column: format!("position {position}"),
                    suggestion: format!(
                        "the schema has {} columns, so positions run from 0 to {}",
                        fields.len(),
                        fields.len().saturating_sub(1)
                    ),
                })
            }
        }
    }
}

fn sk_validate_selected_fields(fields: &[&Field]) -> Result<(), SKError> {
    let columns = fields
        .iter()
        .map(|field| {
            let logical_type =
                sk_logical_type_from_arrow_data_type(field.data_type(), field.name())?;
            Ok(SKColumnSchema::new(
                field.name(),
                logical_type,
                SKColumnRole::Feature,
            ))
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    let _ = SKTabularSchema::new(columns)?;
    Ok(())
}

/// Resolve the selection into the Arrow column positions to read, in order.
///
/// With no selection every column is read in source order. When a selection is
/// supplied, only the selected feature and target columns have their Arrow data
/// types validated.
///
/// # Errors
/// Propagates the schema rejection for an unsupported selected column and the
/// selection errors for an empty schema, unknown column or duplicate selection.
pub(crate) fn sk_projection_indices(
    schema: &Schema,
    selection: Option<&SKColumnSelection>,
) -> Result<Vec<usize>, SKError> {
    let Some(selection) = selection else {
        return Ok((0..schema.fields().len()).collect());
    };
    if schema.fields().is_empty() {
        return Err(SKError::SchemaMismatch {
            expected: "at least one column".to_string(),
            found: "a schema with zero columns".to_string(),
        });
    }
    if selection.feature_columns().is_empty() {
        return Err(SKError::InvalidHyperparameter {
            name: "feature_columns",
            reason: "a column selection must name at least one feature column".to_string(),
        });
    }

    let mut indices =
        Vec::with_capacity(selection.feature_columns().len() + usize::from(selection.has_target()));
    for reference in selection.feature_columns() {
        let index = sk_resolve_reference_in_arrow_schema(schema, reference)?;
        if indices.contains(&index) {
            return Err(SKError::DuplicateColumn {
                column: schema.field(index).name().clone(),
                suggestion: "remove the repeated feature reference".to_string(),
            });
        }
        indices.push(index);
    }

    if let Some(target_reference) = selection.target_column() {
        let target_index = sk_resolve_reference_in_arrow_schema(schema, target_reference)?;
        if indices.contains(&target_index) {
            return Err(SKError::DuplicateColumn {
                column: schema.field(target_index).name().clone(),
                suggestion:
                    "remove the target column from the feature list or choose another target"
                        .to_string(),
            });
        }
        indices.push(target_index);
    }

    let selected_fields: Vec<&Field> = indices.iter().map(|index| schema.field(*index)).collect();
    sk_validate_selected_fields(&selected_fields)?;
    Ok(indices)
}

/// Project a record batch onto the resolved positions, preserving their order.
///
/// # Errors
/// Returns [`SKError::Conversion`] when Arrow rejects the projection.
pub(crate) fn sk_project_record_batch(
    batch: &RecordBatch,
    indices: &[usize],
) -> Result<RecordBatch, SKError> {
    batch
        .project(indices)
        .map_err(|error| SKError::Conversion(format!("Arrow projection failed: {error}")))
}
