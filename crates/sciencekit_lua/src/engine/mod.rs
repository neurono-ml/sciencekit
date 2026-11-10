//! Pure dispatcher: the safe in-process wrappers over the C ABI.
//! Carries no implementation logic.

mod builder;
mod descriptor;
mod family;
mod model;

pub(crate) use builder::SKLuaBuilderHandle;
pub(crate) use family::{SKLuaEstimatorFamily, sk_lua_backend_code};
pub(crate) use model::SKLuaModelHandle;
