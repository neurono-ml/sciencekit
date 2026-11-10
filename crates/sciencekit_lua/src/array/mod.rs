//! Pure dispatcher: the numeric payloads crossing the Lua boundary, their
//! conversion from Lua tables and back. Carries no implementation logic.

mod input;
mod matrix;

#[allow(unused_imports)]
pub(crate) use input::{sk_lua_matrix_from_table, sk_lua_vector_from_table};
pub(crate) use matrix::{SKLuaMatrix, SKLuaVector};

#[cfg(test)]
#[path = "input_tests.rs"]
mod input_tests;
