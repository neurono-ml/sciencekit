//! Pure dispatcher: the safe wrapper over an opaque fitted-model handle.
//! Carries no implementation logic.

mod handle;
mod output;

pub(crate) use handle::SKLuaModelHandle;
