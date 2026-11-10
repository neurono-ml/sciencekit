//! Conversion tests (spec `lua-bindings`, requirement "Array conversion").

use mlua::Lua;

use crate::array::{sk_lua_matrix_from_table, sk_lua_vector_from_table};
use crate::error::SKLuaError;

/// A numeric matrix is copied in row-major order.
#[test]
fn convert_accepts_a_rectangular_matrix() {
    let lua = Lua::new();
    let table: mlua::Table = lua
        .load(r#"return {{1.0, 2.0}, {3, 4.5}, {5.0, 6.0}}"#)
        .eval()
        .expect("the table builds");
    let matrix = sk_lua_matrix_from_table(&table).expect("the matrix converts");
    assert_eq!(matrix.rows(), 3);
    assert_eq!(matrix.columns(), 2);
    assert_eq!(matrix.values(), &[1.0, 2.0, 3.0, 4.5, 5.0, 6.0]);
}

/// A flat sequence converts to a vector.
#[test]
fn convert_accepts_a_flat_vector() {
    let lua = Lua::new();
    let table: mlua::Table = lua.load("return {1.0, 2, 3.5}").eval().unwrap();
    let vector = sk_lua_vector_from_table(&table, 3).expect("the vector converts");
    assert_eq!(vector.values(), &[1.0, 2.0, 3.5]);
}

/// A table of one-element rows is an accepted vector alias.
#[test]
fn convert_accepts_a_column_of_rows_as_a_vector() {
    let lua = Lua::new();
    let table: mlua::Table = lua.load("return {{1.0}, {2.0}, {3.0}}").eval().unwrap();
    let vector = sk_lua_vector_from_table(&table, 3).expect("the vector converts");
    assert_eq!(vector.values(), &[1.0, 2.0, 3.0]);
}

/// Ragged rows are a shape error raised before any ABI call.
#[test]
fn convert_rejects_a_ragged_matrix() {
    let lua = Lua::new();
    let table: mlua::Table = lua.load("return {{1.0, 2.0}, {3.0}}").eval().unwrap();
    let error = sk_lua_matrix_from_table(&table).expect_err("the matrix is ragged");
    assert_eq!(error.status_name(), "shape");
    assert!(error.message().contains("rectangular"));
}

/// A non-numeric cell is an invalid-argument error naming its position.
#[test]
fn convert_rejects_a_non_numeric_cell() {
    let lua = Lua::new();
    let table: mlua::Table = lua.load(r#"return {{1.0, "two"}}"#).eval().unwrap();
    let error = sk_lua_matrix_from_table(&table).expect_err("the cell is not numeric");
    assert_eq!(error.status_name(), "invalid-argument");
    assert!(error.message().contains("row 1, column 2"));
}

/// A row that is not a table is rejected with its index.
#[test]
fn convert_rejects_a_non_table_row() {
    let lua = Lua::new();
    let table: mlua::Table = lua.load(r#"return {{1.0}, 2.0}"#).eval().unwrap();
    let error = sk_lua_matrix_from_table(&table).expect_err("the row is not a table");
    assert_eq!(error.status_name(), "invalid-argument");
    assert!(error.message().contains("row 2"));
}

/// Empty tables are shape errors, not silent zero-row models.
#[test]
fn convert_rejects_empty_tables() {
    let lua = Lua::new();
    let empty: mlua::Table = lua.load("return {}").eval().unwrap();
    let no_columns: mlua::Table = lua.load("return {{}}").eval().unwrap();
    assert!(matches!(
        sk_lua_matrix_from_table(&empty),
        Err(SKLuaError { .. })
    ));
    assert!(sk_lua_matrix_from_table(&no_columns).is_err());
    assert!(sk_lua_vector_from_table(&empty, 0).is_err());
}

/// A target vector must match the number of feature rows.
#[test]
fn convert_rejects_a_vector_length_mismatch() {
    let lua = Lua::new();
    let table: mlua::Table = lua.load("return {1.0, 2.0}").eval().unwrap();
    let error = sk_lua_vector_from_table(&table, 3).expect_err("the length is wrong");
    assert_eq!(error.status_name(), "shape");
    assert!(error.message().contains("expected 3 rows"));
}

/// A target row with more than one value is rejected.
#[test]
fn convert_rejects_a_multi_column_target_row() {
    let lua = Lua::new();
    let table: mlua::Table = lua.load("return {{1.0, 2.0}, {3.0, 4.0}}").eval().unwrap();
    let error = sk_lua_vector_from_table(&table, 2).expect_err("the row is too wide");
    assert_eq!(error.status_name(), "shape");
    assert!(error.message().contains("exactly one value"));
}
