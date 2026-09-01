//! Optional global-allocator selection (spec `allocator-selection`, PRD §4.3).
//!
//! The default build uses the system allocator. The `allocator-jemalloc` and
//! `allocator-mimalloc` features install `tikv-jemallocator` / `mimalloc` as
//! the crate's global allocator; enabling both is rejected at compile time
//! because a single global allocator can be installed at most once.

#[cfg(all(feature = "allocator-jemalloc", feature = "allocator-mimalloc"))]
compile_error!(
    "the `allocator-jemalloc` and `allocator-mimalloc` features are mutually exclusive: \
     enable at most one custom global allocator (or none to keep the system allocator)"
);

#[cfg(feature = "allocator-jemalloc")]
#[global_allocator]
static SK_JEMALLOC_GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "allocator-mimalloc")]
#[global_allocator]
static SK_MIMALLOC_GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The name of the active global allocator, for diagnostics.
pub fn sk_allocator_name() -> &'static str {
    if cfg!(feature = "allocator-jemalloc") {
        "jemalloc"
    } else if cfg!(feature = "allocator-mimalloc") {
        "mimalloc"
    } else {
        "system"
    }
}

#[cfg(test)]
mod allocator_tests;
