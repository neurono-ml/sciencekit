//! Pure dispatcher: the Lua userdata types and their lifecycle.
//! Carries no implementation logic.

mod builder;
mod model;

pub(crate) use builder::sk_lua_builder_userdata;
pub(crate) use model::sk_lua_model_userdata;

#[cfg(test)]
mod gc_tests;
#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod lifecycle_tests;
#[cfg(test)]
#[path = "operations_tests.rs"]
mod operations_tests;
#[cfg(test)]
#[path = "persistence_tests.rs"]
mod persistence_tests;
