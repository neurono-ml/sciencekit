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

/// The bounded set of global allocators this crate can install.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKAllocatorKind {
    /// The platform default allocator (used when no allocator feature is on).
    System,
    /// The `tikv-jemallocator` global allocator (`allocator-jemalloc`).
    Jemalloc,
    /// The `mimalloc` global allocator (`allocator-mimalloc`).
    MiMalloc,
}

impl std::fmt::Display for SKAllocatorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            SKAllocatorKind::System => "system",
            SKAllocatorKind::Jemalloc => "jemalloc",
            SKAllocatorKind::MiMalloc => "mimalloc",
        };
        f.write_str(name)
    }
}

/// The active global allocator, as an enum for diagnostics.
pub fn sk_allocator_kind() -> SKAllocatorKind {
    if cfg!(feature = "allocator-jemalloc") {
        SKAllocatorKind::Jemalloc
    } else if cfg!(feature = "allocator-mimalloc") {
        SKAllocatorKind::MiMalloc
    } else {
        SKAllocatorKind::System
    }
}

#[cfg(test)]
mod allocator_tests;
