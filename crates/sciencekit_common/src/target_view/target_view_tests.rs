//! Tests for the target boundary (spec `data-view-boundary`).

use ndarray::{ArrayView1, array};

use super::SKTargetView;

/// Integer targets elevate losslessly to continuous for a continuous context.
#[test]
fn integer_targets_elevate_for_regression() {
    let ints = array![1_i64, 2, 3];
    let view: SKTargetView<'_, f64> = ArrayView1::from(&ints).try_into().unwrap();
    let continuous = view.as_continuous().unwrap();
    assert_eq!(continuous.as_slice().unwrap(), &[1.0_f64, 2.0, 3.0]);
}

/// Nominal targets reference borrowed text without copying.
#[test]
fn nominal_targets_reference_borrowed_text() {
    let strings: Vec<String> = vec!["cat".into(), "dog".into()];
    let borrowed: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
    let view: SKTargetView<'_, f64> = borrowed.as_slice().try_into().unwrap();
    match view {
        SKTargetView::Nominal(text) => {
            assert_eq!(text, &["cat", "dog"]);
        }
        other => panic!("expected nominal, got {other:?}"),
    }
}

/// Continuous f32 targets pass through in f32 (no f64 detour).
#[test]
fn continuous_f32_targets_stay_f32() {
    let vals = array![0.5_f32, 1.5];
    let view: SKTargetView<'_, f32> = ArrayView1::from(&vals).try_into().unwrap();
    let continuous = view.as_continuous().unwrap();
    assert_eq!(continuous.as_slice().unwrap(), &[0.5_f32, 1.5]);
}

/// Continuous targets pass through as-is (borrowed, no copy).
#[test]
fn continuous_targets_pass_through() {
    let vals = array![0.5_f64, 1.5];
    let view: SKTargetView<'_, f64> = ArrayView1::from(&vals).try_into().unwrap();
    let continuous = view.as_continuous().unwrap();
    assert_eq!(continuous.as_slice().unwrap(), &[0.5, 1.5]);
}

/// Nominal targets are not continuous — elevation is rejected precisely.
#[test]
fn nominal_is_not_continuous() {
    let borrowed: Vec<&str> = vec!["a", "b"];
    let view: SKTargetView<'_, f64> = borrowed.as_slice().try_into().unwrap();
    assert!(view.as_continuous().is_err());
}
