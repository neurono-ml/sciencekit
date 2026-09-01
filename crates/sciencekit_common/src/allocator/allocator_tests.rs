//! Tests for the allocator selection (spec `allocator-selection`).
//!
//! The both-enabled conflict guard (`compile_error!`) cannot be exercised from
//! a unit test; it is verified by building with both allocator features
//! enabled, which must fail with the conflict message (see `mod.rs`).

use super::sk_allocator_name;

/// The default build uses the system allocator and still allocates fine.
#[cfg(not(any(feature = "allocator-jemalloc", feature = "allocator-mimalloc")))]
#[test]
fn default_build_uses_the_system_allocator() {
    assert_eq!(sk_allocator_name(), "system");
    let values = vec![0u8; 4096];
    assert_eq!(values.len(), 4096);
}

/// The `allocator-jemalloc` feature installs tikv-jemallocator.
#[cfg(feature = "allocator-jemalloc")]
#[test]
fn jemalloc_feature_installs_jemalloc() {
    assert_eq!(sk_allocator_name(), "jemalloc");
    let values: Vec<u64> = (0..2048).map(|index| index as u64).collect();
    assert_eq!(values.len(), 2048);
    assert_eq!(values[2047], 2047);
}

/// The `allocator-mimalloc` feature installs mimalloc.
#[cfg(feature = "allocator-mimalloc")]
#[test]
fn mimalloc_feature_installs_mimalloc() {
    assert_eq!(sk_allocator_name(), "mimalloc");
    let values: Vec<u64> = (0..2048).map(|index| index as u64).collect();
    assert_eq!(values.len(), 2048);
    assert_eq!(values[2047], 2047);
}
