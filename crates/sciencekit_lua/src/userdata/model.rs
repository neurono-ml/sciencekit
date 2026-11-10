//! The fitted-model userdata: operations, fields and `close`.
//!
//! Like the builder, the model releases its opaque handle through `close`,
//! the `__close` metamethod and the engine wrapper's `Drop` — the hook Lua's
//! garbage collector triggers. The three paths are idempotent.

use mlua::{AnyUserData, Lua, MetaMethod, UserData, UserDataFields, UserDataMethods, Value};

use crate::engine::SKLuaEstimatorFamily;
use crate::engine::SKLuaModelHandle;

/// A fitted model exposed to Lua.
#[derive(Debug)]
pub(crate) struct SKLuaModel {
    pub(crate) handle: SKLuaModelHandle,
    pub(crate) family: SKLuaEstimatorFamily,
}

/// Wrap an engine handle in a Lua userdata.
pub(crate) fn sk_lua_model_userdata(
    lua: &Lua,
    handle: SKLuaModelHandle,
    family: SKLuaEstimatorFamily,
) -> mlua::Result<AnyUserData> {
    lua.create_userdata(SKLuaModel { handle, family })
}

impl UserData for SKLuaModel {
    fn add_fields<F: UserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("family", |_, this| Ok(this.family.name()));
        fields.add_field_method_get("dtype", |_, _| Ok("f64"));
        fields.add_field_method_get("closed", |_, this| Ok(this.handle.is_closed()));
    }

    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("family", |_, this, ()| Ok(this.family.name()));
        methods.add_method("dtype", |_, _, ()| Ok("f64"));
        methods.add_method("closed", |_, this, ()| Ok(this.handle.is_closed()));
        methods.add_method_mut("close", |_, this, ()| {
            this.handle.close();
            Ok(())
        });
        methods.add_meta_method_mut(MetaMethod::Close, |_, this, _error: Value| {
            this.handle.close();
            Ok(())
        });
        methods.add_meta_method(MetaMethod::ToString, |_, this, ()| {
            Ok(format!(
                "sciencekit model ({}){}",
                this.family.name(),
                if this.handle.is_closed() {
                    " [closed]"
                } else {
                    ""
                }
            ))
        });
    }
}
