//! The builder userdata: chainable setters, fields and `close`.
//!
//! Lua never sees the raw handle. `close` and the `__close` metamethod both
//! forward to the engine wrapper, whose `Drop` implementation is what Lua's
//! garbage collector invokes when the userdata becomes unreachable. All three
//! paths are idempotent, so a handle is released exactly once.

use mlua::{AnyUserData, Lua, MetaMethod, Table, UserData, UserDataFields, UserDataMethods, Value};

use crate::array::{sk_lua_matrix_from_table, sk_lua_vector_from_table};
use crate::engine::SKLuaBuilderHandle;
use crate::engine::{SKLuaEstimatorFamily, sk_lua_backend_code};
use crate::error::SKLuaError;
use crate::userdata::sk_lua_model_userdata;

/// A configured-but-not-fitted estimator exposed to Lua.
#[derive(Debug)]
pub(crate) struct SKLuaBuilder {
    pub(crate) handle: SKLuaBuilderHandle,
    pub(crate) family: SKLuaEstimatorFamily,
}

/// Create the userdata for one family.
pub(crate) fn sk_lua_builder_userdata(
    lua: &Lua,
    family: SKLuaEstimatorFamily,
) -> mlua::Result<AnyUserData> {
    let handle = SKLuaBuilderHandle::create(family)?;
    lua.create_userdata(SKLuaBuilder { handle, family })
}

impl UserData for SKLuaBuilder {
    fn add_fields<F: UserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("family", |_, this| Ok(this.family.name()));
        fields.add_field_method_get("dtype", |_, _| Ok("f64"));
        fields.add_field_method_get("closed", |_, this| Ok(this.handle.is_closed()));
    }

    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_function_mut(
            "include_intercept",
            |_, (this, enabled): (AnyUserData, bool)| {
                this.borrow_mut::<SKLuaBuilder>()?
                    .handle
                    .set_include_intercept(enabled)?;
                Ok(this)
            },
        );
        methods.add_function_mut("with_mean", |_, (this, enabled): (AnyUserData, bool)| {
            this.borrow_mut::<SKLuaBuilder>()?
                .handle
                .set_with_mean(enabled)?;
            Ok(this)
        });
        methods.add_function_mut("with_std", |_, (this, enabled): (AnyUserData, bool)| {
            this.borrow_mut::<SKLuaBuilder>()?
                .handle
                .set_with_std(enabled)?;
            Ok(this)
        });
        methods.add_function_mut("backend", |_, (this, backend): (AnyUserData, Value)| {
            let code = sk_lua_backend_code(&backend)?;
            this.borrow_mut::<SKLuaBuilder>()?
                .handle
                .set_backend(code)?;
            Ok(this)
        });
        methods.add_method(
            "fit",
            |lua, this, (features, targets): (Table, Option<Table>)| {
                let features = sk_lua_matrix_from_table(&features)?;
                let targets = match targets {
                    Some(targets) => Some(sk_lua_vector_from_table(&targets, features.rows())?),
                    None if this.family == SKLuaEstimatorFamily::LinearRegression => {
                        return Err(SKLuaError::invalid_argument(
                            "linear regression requires targets",
                        )
                        .into());
                    }
                    None => None,
                };
                let model = this.handle.fit(&features, targets.as_ref())?;
                sk_lua_model_userdata(lua, model, this.family)
            },
        );
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
                "sciencekit builder ({}){}",
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
