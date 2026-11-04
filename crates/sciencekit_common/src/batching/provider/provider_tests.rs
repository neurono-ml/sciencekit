//! Tests for the feature-only source provider (spec
//! `streaming-source-provider`).

use std::cell::Cell;
use std::rc::Rc;

use super::provider_test_sources::{FiniteSource, drain_feature_positions};
use super::{SKReopenableSource, SKSourceProvider};
use crate::{SKError, SKFloat};

fn provider_is_reopenable<F, Provider: SKSourceProvider<F>>(_: &Provider) -> bool {
    Provider::IS_REOPENABLE
}

fn one_shot_provider_yields_once<F: SKFloat>() {
    let mut source = FiniteSource::<F>::new(3);
    let mut provider = &mut source;
    assert!(!provider_is_reopenable::<F, _>(&provider));

    {
        let opened = SKSourceProvider::open(&mut provider).unwrap();
        assert_eq!(drain_feature_positions(opened), vec![0, 1, 2]);
    }
    {
        let reopened = SKSourceProvider::open(&mut provider).unwrap();
        assert!(drain_feature_positions(reopened).is_empty());
    }
}

fn reopenable_factory_is_invoked_once_per_pass<F: SKFloat>() {
    let opens = Rc::new(Cell::new(0));
    let counter = Rc::clone(&opens);
    let mut provider = SKReopenableSource::new(move || {
        counter.set(counter.get() + 1);
        Ok(FiniteSource::<F>::new(3))
    });
    assert!(provider_is_reopenable::<F, _>(&provider));

    for _pass in 0..2 {
        let mut source = SKSourceProvider::open(&mut provider).unwrap();
        assert_eq!(drain_feature_positions(&mut source), vec![0, 1, 2]);
    }
    assert_eq!(opens.get(), 2);
}

fn factory_error_propagates<F: SKFloat>() {
    let mut calls = 0usize;
    let mut provider = SKReopenableSource::new(move || {
        calls += 1;
        if calls == 2 {
            Err(SKError::Conversion(
                "factory failed on the second pass".into(),
            ))
        } else {
            Ok(FiniteSource::<F>::new(2))
        }
    });

    {
        let mut first = SKSourceProvider::open(&mut provider).unwrap();
        assert_eq!(drain_feature_positions(&mut first), vec![0, 1]);
    }
    let error = match SKSourceProvider::open(&mut provider) {
        Ok(_) => panic!("the factory must fail on the second pass"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("factory failed on the second pass")
    );
}

#[test]
fn one_shot_provider_yields_once_f64() {
    one_shot_provider_yields_once::<f64>();
}

#[test]
fn one_shot_provider_yields_once_f32() {
    one_shot_provider_yields_once::<f32>();
}

#[test]
fn reopenable_factory_is_invoked_once_per_pass_f64() {
    reopenable_factory_is_invoked_once_per_pass::<f64>();
}

#[test]
fn reopenable_factory_is_invoked_once_per_pass_f32() {
    reopenable_factory_is_invoked_once_per_pass::<f32>();
}

#[test]
fn factory_error_propagates_f64() {
    factory_error_propagates::<f64>();
}

#[test]
fn factory_error_propagates_f32() {
    factory_error_propagates::<f32>();
}
