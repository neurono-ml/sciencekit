//! Lua table to numeric buffer conversion (spec `lua-bindings`, requirement
//! "Array conversion").
//!
//! A matrix is a table of equal-length rows; a vector is a flat sequence of
//! numbers (a table of one-element rows is accepted as an alias). Every
//! table is copied into a contiguous `f64` buffer. Validation happens before
//! any ABI call, so a ragged table never reaches the library.

use mlua::{Table, Value};

use crate::error::SKLuaError;

use super::{SKLuaMatrix, SKLuaVector};

/// Copy a table of equal-length rows into a matrix.
pub(crate) fn sk_lua_matrix_from_table(table: &Table) -> Result<SKLuaMatrix, SKLuaError> {
    let rows = table.raw_len();
    if rows == 0 {
        return Err(SKLuaError::shape("features must contain at least one row"));
    }
    let columns = sk_lua_row_table(table, 1)?.raw_len();
    if columns == 0 {
        return Err(SKLuaError::shape(
            "feature rows must contain at least one value",
        ));
    }
    let mut values = Vec::with_capacity(rows * columns);
    for row_index in 1..=rows {
        let row = sk_lua_row_table(table, row_index)?;
        let length = row.raw_len();
        if length != columns {
            return Err(SKLuaError::shape(format!(
                "feature row {row_index} has {length} values, expected {columns}; \
                 a matrix must be rectangular"
            )));
        }
        for column_index in 1..=columns {
            values.push(sk_lua_number(&row, row_index, column_index)?);
        }
    }
    Ok(SKLuaMatrix::new(rows, columns, values))
}

/// Copy a vector from a flat table or a table of one-element rows.
pub(crate) fn sk_lua_vector_from_table(
    table: &Table,
    expected_length: usize,
) -> Result<SKLuaVector, SKLuaError> {
    let length = table.raw_len();
    if length == 0 {
        return Err(SKLuaError::shape("targets must contain at least one value"));
    }
    if length != expected_length {
        return Err(SKLuaError::shape(format!(
            "targets has {length} values, expected {expected_length} rows"
        )));
    }
    let first = table.get::<Value>(1).map_err(SKLuaError::from)?;
    let mut values = Vec::with_capacity(length);
    if matches!(first, Value::Table(_)) {
        for row_index in 1..=length {
            let row = sk_lua_row_table(table, row_index)?;
            if row.raw_len() != 1 {
                return Err(SKLuaError::shape(format!(
                    "target row {row_index} must contain exactly one value"
                )));
            }
            values.push(sk_lua_number(&row, row_index, 1)?);
        }
    } else {
        for index in 1..=length {
            values.push(sk_lua_number(table, 1, index)?);
        }
    }
    Ok(SKLuaVector::new(values))
}

/// Read one row, rejecting a cell that is not itself a table.
fn sk_lua_row_table(table: &Table, row_index: usize) -> Result<Table, SKLuaError> {
    match table.get::<Value>(row_index).map_err(SKLuaError::from)? {
        Value::Table(row) => Ok(row),
        other => Err(SKLuaError::invalid_argument(format!(
            "row {row_index} must be a table of numbers, found {}",
            other.type_name()
        ))),
    }
}

/// Read one numeric cell, accepting both Lua floats and integers.
fn sk_lua_number(table: &Table, row_index: usize, column_index: usize) -> Result<f64, SKLuaError> {
    match table.get::<Value>(column_index).map_err(SKLuaError::from)? {
        Value::Number(value) => Ok(value),
        Value::Integer(value) => Ok(value as f64),
        other => Err(SKLuaError::invalid_argument(format!(
            "value at row {row_index}, column {column_index} must be a number, found {}",
            other.type_name()
        ))),
    }
}
