//! A column-major, typed batch for the width of one streaming block (spec
//! `dataset-interop-core`).

use sciencekit_common::SKError;

use crate::columns::SKColumnData;
use crate::schema::{SKColumnSchema, SKTabularSchema};

/// A heterogeneous, column-major table block.
///
/// The batch pairs the schema that describes its columns with the typed column
/// data, its zero-based position in the source and whether it is the final
/// block. Construction enforces that every column is row-aligned with the
/// schema, so no conversion ever meets a ragged table.
#[derive(Debug, Clone, PartialEq)]
pub struct SKTabularBatch {
    schema: SKTabularSchema,
    columns: Vec<SKColumnData>,
    position: usize,
    is_final: bool,
}

impl SKTabularBatch {
    /// Build a batch from a schema, its columns and streaming metadata.
    ///
    /// # Errors
    /// Returns [`SKError::SchemaMismatch`] when the column count or a column's
    /// logical type differs from the schema, and
    /// [`SKError::UnevenColumnLengths`] when a column's length differs from the
    /// batch's row count.
    pub fn new(
        schema: SKTabularSchema,
        columns: Vec<SKColumnData>,
        position: usize,
        is_final: bool,
    ) -> Result<Self, SKError> {
        if columns.len() != schema.len() {
            return Err(SKError::SchemaMismatch {
                expected: format!("{} columns", schema.len()),
                found: format!("{} columns", columns.len()),
            });
        }

        for (column_schema, column_data) in schema.columns().iter().zip(&columns) {
            if column_schema.logical_type() != column_data.logical_type() {
                return Err(SKError::SchemaMismatch {
                    expected: format!(
                        "column `{}` of logical type {}",
                        column_schema.name(),
                        column_schema.logical_type()
                    ),
                    found: column_data.logical_type().to_string(),
                });
            }
        }

        let row_count = columns.first().map_or(0, SKColumnData::len);
        for (column_schema, column_data) in schema.columns().iter().zip(&columns) {
            if column_data.len() != row_count {
                return Err(SKError::UnevenColumnLengths {
                    column: column_schema.name().to_string(),
                    expected: row_count,
                    found: column_data.len(),
                });
            }
        }

        Ok(SKTabularBatch {
            schema,
            columns,
            position,
            is_final,
        })
    }

    /// The schema describing the batch's columns.
    pub fn schema(&self) -> &SKTabularSchema {
        &self.schema
    }

    /// The typed column data in schema order.
    pub fn columns(&self) -> &[SKColumnData] {
        &self.columns
    }

    /// The column data at a zero-based position.
    pub fn column(&self, position: usize) -> Option<&SKColumnData> {
        self.columns.get(position)
    }

    /// The column data with the given schema name.
    pub fn column_named(&self, name: &str) -> Option<&SKColumnData> {
        self.schema
            .column_index(name)
            .and_then(|index| self.columns.get(index))
    }

    /// The number of rows; zero for an empty batch.
    pub fn number_of_rows(&self) -> usize {
        self.columns.first().map_or(0, SKColumnData::len)
    }

    /// The number of columns.
    pub fn number_of_columns(&self) -> usize {
        self.columns.len()
    }

    /// Whether the batch carries no row.
    pub fn is_empty(&self) -> bool {
        self.number_of_rows() == 0
    }

    /// The zero-based position of this batch in the source.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether this is the final block of a finite source.
    pub fn is_final(&self) -> bool {
        self.is_final
    }

    /// Project the batch onto a subset of columns, in the given order.
    ///
    /// # Errors
    /// Returns [`SKError::UnknownColumn`] for an out-of-range position and
    /// [`SKError::DuplicateColumn`] when a column is projected twice.
    pub fn project(&self, positions: &[usize]) -> Result<Self, SKError> {
        let mut projected_schema = Vec::with_capacity(positions.len());
        let mut projected_columns = Vec::with_capacity(positions.len());
        for position in positions {
            let column_schema =
                self.schema
                    .column_at(*position)
                    .ok_or_else(|| SKError::UnknownColumn {
                        column: format!("position {position}"),
                        suggestion: format!(
                            "the batch has {} columns, so positions run from 0 to {}",
                            self.schema.len(),
                            self.schema.len().saturating_sub(1)
                        ),
                    })?;
            if projected_schema
                .iter()
                .any(|existing: &SKColumnSchema| existing.name() == column_schema.name())
            {
                return Err(SKError::DuplicateColumn {
                    column: column_schema.name().to_string(),
                    suggestion: "project each column at most once".to_string(),
                });
            }
            projected_schema.push(column_schema.clone());
            projected_columns.push(self.columns[*position].clone());
        }

        let schema = SKTabularSchema::new(projected_schema)?;
        SKTabularBatch::new(schema, projected_columns, self.position, self.is_final)
    }
}
