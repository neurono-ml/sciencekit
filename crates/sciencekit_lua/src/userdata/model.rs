//! The fitted-model userdata: operations, fields and `close`.
//!
//! Like the builder, the model releases its opaque handle through `close`,
//! the `__close` metamethod and the engine wrapper's `Drop` — the hook Lua's
//! garbage collector triggers. The three paths are idempotent.

use mlua::{AnyUserData, Lua, MetaMethod, Table, UserData, UserDataFields, UserDataMethods, Value};

use crate::array::{sk_lua_matrix_from_table, sk_lua_table_from_matrix, sk_lua_vector_from_table};
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
        methods.add_method("predict", |lua, this, features: Table| {
            let features = sk_lua_matrix_from_table(&features)?;
            let predictions = this.handle.predict(&features)?;
            Ok(sk_lua_table_from_matrix(lua, &predictions)?)
        });
        methods.add_method("transform", |lua, this, features: Table| {
            let features = sk_lua_matrix_from_table(&features)?;
            let transformed = this.handle.transform(&features)?;
            Ok(sk_lua_table_from_matrix(lua, &transformed)?)
        });
        methods.add_method("score", |_, this, (features, targets): (Table, Table)| {
            let features = sk_lua_matrix_from_table(&features)?;
            let targets = sk_lua_vector_from_table(&targets, features.rows())?;
            Ok(this.handle.score(&features, &targets)?)
        });
        methods.add_method("save", |_, this, path: String| {
            this.handle.save(&path)?;
            Ok(())
        });
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
