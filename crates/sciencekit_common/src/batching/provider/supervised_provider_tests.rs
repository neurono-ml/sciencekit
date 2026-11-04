//! Tests for the supervised source provider and the agreement between both
//! families (spec `streaming-source-provider`).

use std::cell::Cell;
use std::rc::Rc;

use super::provider_test_sources::{
    FiniteSource, drain_feature_positions, drain_supervised_positions,
};
use super::{SKReopenableSource, SKSourceProvider, SKSupervisedSourceProvider};
use crate::SKFloat;

fn supervised_provider_is_reopenable<F, Provider: SKSupervisedSourceProvider<F>>(
    _: &Provider,
) -> bool {
    Provider::IS_REOPENABLE
}

fn supervised_one_shot_provider_yields_once<F: SKFloat>() {
    let mut source = FiniteSource::<F>::new(3);
    let mut provider = &mut source;
    assert!(!supervised_provider_is_reopenable::<F, _>(&provider));

    {
        let opened = SKSupervisedSourceProvider::open(&mut provider).unwrap();
        assert_eq!(drain_supervised_positions(opened), vec![0, 1, 2]);
    }
    {
        let reopened = SKSupervisedSourceProvider::open(&mut provider).unwrap();
        assert!(drain_supervised_positions(reopened).is_empty());
    }
}

fn supervised_reopenable_factory_is_invoked_once_per_pass<F: SKFloat>() {
    let opens = Rc::new(Cell::new(0));
    let counter = Rc::clone(&opens);
    let mut provider = SKReopenableSource::new(move || {
        counter.set(counter.get() + 1);
        Ok(FiniteSource::<F>::new(3))
    });
    assert!(supervised_provider_is_reopenable::<F, _>(&provider));

    for _pass in 0..2 {
        let mut source = SKSupervisedSourceProvider::open(&mut provider).unwrap();
        assert_eq!(drain_supervised_positions(&mut source), vec![0, 1, 2]);
    }
    assert_eq!(opens.get(), 2);
}

fn families_agree_on_passes_and_order<F: SKFloat>() {
    let feature_opens = Rc::new(Cell::new(0));
    let feature_counter = Rc::clone(&feature_opens);
    let mut feature_provider = SKReopenableSource::new(move || {
        feature_counter.set(feature_counter.get() + 1);
        Ok(FiniteSource::<F>::new(3))
    });
    let supervised_opens = Rc::new(Cell::new(0));
    let supervised_counter = Rc::clone(&supervised_opens);
    let mut supervised_provider = SKReopenableSource::new(move || {
        supervised_counter.set(supervised_counter.get() + 1);
        Ok(FiniteSource::<F>::new(3))
    });

    let mut feature_passes = Vec::new();
    for _pass in 0..2 {
        let mut source = SKSourceProvider::open(&mut feature_provider).unwrap();
        feature_passes.push(drain_feature_positions(&mut source));
    }
    let mut supervised_passes = Vec::new();
    for _pass in 0..2 {
        let mut source = SKSupervisedSourceProvider::open(&mut supervised_provider).unwrap();
        supervised_passes.push(drain_supervised_positions(&mut source));
    }

    assert_eq!(feature_passes, supervised_passes);
    assert_eq!(feature_opens.get(), supervised_opens.get());
}

#[test]
fn supervised_one_shot_provider_yields_once_f64() {
    supervised_one_shot_provider_yields_once::<f64>();
}

#[test]
fn supervised_one_shot_provider_yields_once_f32() {
    supervised_one_shot_provider_yields_once::<f32>();
}

#[test]
fn supervised_reopenable_factory_is_invoked_once_per_pass_f64() {
    supervised_reopenable_factory_is_invoked_once_per_pass::<f64>();
}

#[test]
fn supervised_reopenable_factory_is_invoked_once_per_pass_f32() {
    supervised_reopenable_factory_is_invoked_once_per_pass::<f32>();
}

#[test]
fn families_agree_on_passes_and_order_f64() {
    families_agree_on_passes_and_order::<f64>();
}

#[test]
fn families_agree_on_passes_and_order_f32() {
    families_agree_on_passes_and_order::<f32>();
}
