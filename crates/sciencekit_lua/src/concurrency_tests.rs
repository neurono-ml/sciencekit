//! Concurrent-use tests (spec `lua-bindings`, acceptance: "runs ... under
//! concurrency").
//!
//! The C ABI documents fitted models as immutable and safe to read from
//! several threads, and the handle registry hands every operation its own
//! `Arc`. mlua's `Lua` is deliberately not `Send` in module mode, so each Lua
//! worker builds its own VM; the shared-model test drives the engine wrapper
//! directly from several threads.

use std::sync::Arc;
use std::thread;

use mlua::Lua;

use crate::array::{SKLuaMatrix, SKLuaVector};
use crate::engine::SKLuaBuilderHandle;
use crate::engine::SKLuaEstimatorFamily;
use crate::testing::sk_lua_test_environment;

/// Several threads predict from one shared model and agree exactly.
#[test]
fn concurrency_shared_model_predicts_deterministically() {
    let features = SKLuaMatrix::new(4, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    let targets = SKLuaVector::new(vec![1.0, 3.0, 4.0, 14.0]);
    let model = Arc::new({
        let builder =
            SKLuaBuilderHandle::create(SKLuaEstimatorFamily::LinearRegression).expect("builder");
        builder.fit(&features, Some(&targets)).expect("fit")
    });
    let expected = model.predict(&features).expect("the baseline prediction");
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);
            let features = features.clone();
            let expected = expected.clone();
            thread::spawn(move || {
                for _ in 0..25 {
                    let predictions = model.predict(&features).expect("prediction");
                    assert_eq!(predictions, expected);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().expect("no worker may panic");
    }
}

/// Each thread owns an embedded VM, fits its own model and frees it on drop.
#[test]
fn concurrency_independent_vms_fit_and_predict() {
    let workers: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let lua = sk_lua_test_environment();
                let prediction: f64 = lua
                    .load(
                        r#"
                        local sciencekit = require("sciencekit")
                        local features = {{1.0}, {2.0}, {3.0}}
                        local targets = {2.0, 4.0, 6.0}
                        local model = sciencekit.linear_regression():fit(features, targets)
                        return model:predict({{4.0}})[1][1]
                    "#,
                    )
                    .eval()
                    .expect("the worker fits and predicts");
                prediction
            })
        })
        .collect();
    for worker in workers {
        let prediction = worker.join().expect("no worker may panic");
        assert!(
            (prediction - 8.0).abs() < 1.0e-8,
            "unexpected prediction {prediction}"
        );
    }
}

/// A worker can free its handles while another thread keeps predicting from a
/// distinct model; the registry must not interfere.
#[test]
fn concurrency_freeing_one_model_does_not_disturb_another() {
    let features = SKLuaMatrix::new(3, 1, vec![1.0, 2.0, 3.0]);
    let targets = SKLuaVector::new(vec![2.0, 4.0, 6.0]);
    let keeper = {
        let builder = SKLuaBuilderHandle::create(SKLuaEstimatorFamily::LinearRegression).unwrap();
        Arc::new(builder.fit(&features, Some(&targets)).unwrap())
    };
    let short_lived = {
        let builder = SKLuaBuilderHandle::create(SKLuaEstimatorFamily::LinearRegression).unwrap();
        builder.fit(&features, Some(&targets)).unwrap()
    };
    let reader = {
        let keeper = Arc::clone(&keeper);
        let features = features.clone();
        thread::spawn(move || {
            for _ in 0..50 {
                keeper
                    .predict(&features)
                    .expect("the keeper still predicts");
            }
        })
    };
    drop(short_lived);
    reader.join().expect("no worker may panic");
    // A second VM can still load the module after the handles changed.
    let _lua: Lua = sk_lua_test_environment();
}
