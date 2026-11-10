//! Pure dispatcher: the numeric payloads crossing the Lua boundary.
//! Carries no implementation logic.

mod matrix;

pub(crate) use matrix::{SKLuaMatrix, SKLuaVector};
