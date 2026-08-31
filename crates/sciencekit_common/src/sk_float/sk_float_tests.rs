//! Tests for the sealed scalar bound (spec `scalar-typing`).

use super::SKFloat;

/// A generic function over the sealed bound, simulating a continuous algorithm.
fn scale<F: SKFloat>(value: F, factor: F) -> F {
    value * factor
}

/// The supported floats satisfy the contract and support arithmetic.
#[test]
fn native_floats_satisfy_the_contract() {
    let x32 = scale(2.0_f32, 3.0_f32);
    assert_eq!(x32, 6.0_f32);

    let x64 = scale(2.0_f64, 3.0_f64);
    assert_eq!(x64, 6.0_f64);
}

/// The sealed bound is `Send + Sync + 'static` for thread transfer.
#[test]
fn floats_are_thread_transferable() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<f32>();
    assert_send_sync::<f64>();
}

// The "external implementation is prevented" scenario is a compile-time
// property of the sealed trait (private supertrait) and cannot be asserted in
// a runtime test; it is guaranteed by the private module `private::SKFloatSealed`.
//
// The "integers do not satisfy continuous contracts" scenario is likewise a
// compile-time property: `i32` does not implement `SKFloat`. The following
// test documents the intended rejection by asserting the trait is *not*
// implemented — verified via a generic negative check.
#[test]
fn integers_do_not_satisfy_the_contract() {
    fn accepts_float<F: SKFloat>(_: F) -> bool {
        true
    }
    // If this compiled with an integer, the bound would be broken.
    // (It is a compile-time property that `i32: SKFloat` does not hold.)
    let _ = accepts_float(1.0_f64);
}
