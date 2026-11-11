//! Arrow schema ↔ neutral schema mapping in both directions (spec
//! `arrow-columnar-interop`).

use std::collections::HashMap;
use std::str::FromStr;

use arrow_schema::{DataType, Field, Schema};
use sciencekit_common::SKError;

use super::data_type_mapping::sk_logical_type_from_arrow_data_type;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

/// Field metadata key carrying the column role across a round trip.
pub const SK_ARROW_ROLE_METADATA_KEY: &str = "sciencekit:role";

/// Field metadata key carrying the neutral logical type across a round trip.
///
/// It is redundant for floating, boolean and text columns but disambiguates
/// [`SKLogicalType::CategoryCode`] from [`SKLogicalType::Integer64`], which both
/// map onto Arrow's `Int64`.
pub const SK_ARROW_LOGICAL_TYPE_METADATA_KEY: &str = "sciencekit:logical_type";

/// The role assigned to each Arrow column by name.
///
/// Columns without an entry default to [`SKColumnRole::Feature`]. An explicit
/// `sciencekit:role` field metadata entry wins over this map, so a schema
/// exported by sciencekit round-trips its roles even without a request.
#[derive(Debug, Clone, Default)]
pub struct SKArrowColumnRoles {
    entries: Vec<(String, SKColumnRole)>,
}

impl SKArrowColumnRoles {
    /// An empty role assignment: every column is a feature.
    pub fn new() -> Self {
        SKArrowColumnRoles::default()
    }

    /// Assign a role to a column, replacing a previous assignment.
    pub fn role(mut self, name: impl Into<String>, role: SKColumnRole) -> Self {
        let name = name.into();
        self.entries.retain(|(existing, _)| existing != &name);
        self.entries.push((name, role));
        self
    }

    /// Mark a column as the supervised target.
    pub fn target(self, name: impl Into<String>) -> Self {
        self.role(name, SKColumnRole::Target)
    }

    /// The requested role of a column, defaulting to feature.
    pub fn role_of(&self, name: &str) -> SKColumnRole {
        self.entries
            .iter()
            .find(|(existing, _)| existing == name)
            .map(|(_, role)| *role)
            .unwrap_or(SKColumnRole::Feature)
    }
}

/// Resolve the role of one Arrow field: metadata first, request second.
fn sk_role_of_field(field: &Field, roles: &SKArrowColumnRoles) -> SKColumnRole {
    field
        .metadata()
        .get(SK_ARROW_ROLE_METADATA_KEY)
        .and_then(|raw| SKColumnRole::from_str(raw).ok())
        .unwrap_or_else(|| roles.role_of(field.name()))
}

/// Resolve the logical type of one Arrow field, honoring an exported metadata
/// declaration when it is compatible with the physical type.
///
/// The metadata only matters for [`SKLogicalType::CategoryCode`], which shares
/// Arrow's `Int64` with [`SKLogicalType::Integer64`]; any other declaration is
/// ignored so a stale annotation cannot change the physical reading.
fn sk_logical_type_of_field(field: &Field) -> Result<SKLogicalType, SKError> {
    let inferred = sk_logical_type_from_arrow_data_type(field.data_type(), field.name())?;
    let declared = field
        .metadata()
        .get(SK_ARROW_LOGICAL_TYPE_METADATA_KEY)
        .and_then(|raw| SKLogicalType::from_name(raw).ok());
    match declared {
        Some(SKLogicalType::CategoryCode) if inferred == SKLogicalType::Integer64 => {
            Ok(SKLogicalType::CategoryCode)
        }
        Some(declared) if declared == inferred => Ok(declared),
        _ => Ok(inferred),
    }
}

/// Map an Arrow schema onto a validated neutral schema.
///
/// Column names and order are preserved verbatim; duplicate names are rejected
/// by [`SKTabularSchema::new`].
///
/// # Errors
/// Propagates [`SKError::UnsupportedColumnType`] for a type with no neutral
/// counterpart and the schema-construction errors for empty or duplicated
/// columns.
pub fn sk_tabular_schema_from_arrow_schema(
    schema: &Schema,
    roles: &SKArrowColumnRoles,
) -> Result<SKTabularSchema, SKError> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            let logical_type = sk_logical_type_of_field(field)?;
            Ok(SKColumnSchema::new(
                field.name(),
                logical_type,
                sk_role_of_field(field, roles),
            ))
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    SKTabularSchema::new(columns)
}

/// Map one neutral column onto its Arrow data type.
///
/// # Errors
/// Returns [`SKError::UnsupportedColumnType`] when the logical type has no Arrow
/// counterpart, which is unreachable for the closed neutral set but kept as a
/// structured failure rather than a panic.
pub fn sk_arrow_data_type_from_logical_type(
    logical_type: SKLogicalType,
) -> Result<DataType, SKError> {
    match logical_type {
        SKLogicalType::Float32 => Ok(DataType::Float32),
        SKLogicalType::Float64 => Ok(DataType::Float64),
        SKLogicalType::Integer64 | SKLogicalType::CategoryCode => Ok(DataType::Int64),
        SKLogicalType::Boolean => Ok(DataType::Boolean),
        SKLogicalType::Nominal => Ok(DataType::Utf8),
    }
}

/// Map a neutral schema onto an Arrow schema, carrying roles and logical types
/// as field metadata.
///
/// Every exported field is nullable: the neutral schema does not record
/// nullability, so the safe declaration is the permissive one.
///
/// # Errors
/// Propagates the logical-type mapping failure of
/// [`sk_arrow_data_type_from_logical_type`].
pub fn sk_arrow_schema_from_tabular_schema(schema: &SKTabularSchema) -> Result<Schema, SKError> {
    let fields = schema
        .columns()
        .iter()
        .map(|column| {
            let data_type = sk_arrow_data_type_from_logical_type(column.logical_type())?;
            let metadata = HashMap::from([
                (
                    SK_ARROW_ROLE_METADATA_KEY.to_string(),
                    column.role().name().to_string(),
                ),
                (
                    SK_ARROW_LOGICAL_TYPE_METADATA_KEY.to_string(),
                    column.logical_type().name().to_string(),
                ),
            ]);
            Ok(Field::new(column.name(), data_type, true).with_metadata(metadata))
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    Ok(Schema::new(fields))
}
