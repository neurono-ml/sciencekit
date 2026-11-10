//! Pure dispatcher: the Lua userdata types and their lifecycle.
//! Carries no implementation logic.

mod builder;
mod model;

pub(crate) use builder::sk_lua_builder_userdata;
pub(crate) use model::sk_lua_model_userdata;

#[cfg(test)]
mod lifecycle_tests;
