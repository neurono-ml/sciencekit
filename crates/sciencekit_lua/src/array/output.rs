//! Numeric buffer to Lua table conversion.
//!
//! Outputs are fresh tables of rows: predictions are one-element rows and
//! transforms keep their column count. The copy is explicit; nothing in Lua
//! aliases the ABI buffers, which are released before the table is returned.

use mlua::{Lua, Table};

use crate::error::SKLuaError;

use super::SKLuaMatrix;

/// Build a table of rows from a matrix.
pub(crate) fn sk_lua_table_from_matrix(
    lua: &Lua,
    matrix: &SKLuaMatrix,
) -> Result<Table, SKLuaError> {
    let table = lua.create_table_with_capacity(matrix.rows(), 0)?;
    for row_index in 0..matrix.rows() {
        let row = lua.create_table_with_capacity(matrix.columns(), 0)?;
        for column_index in 0..matrix.columns() {
            let value = matrix.values()[row_index * matrix.columns() + column_index];
            row.set(column_index + 1, value)?;
        }
        table.set(row_index + 1, row)?;
    }
    Ok(table)
}
