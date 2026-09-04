//! Optional global-allocator selection (spec `allocator-selection`, PRD §4.3).
//!
//! The default build uses the system allocator. The `allocator-jemalloc` and
//! `allocator-mimalloc` features install `tikv-jemallocator` / `mimalloc` as
//! the crate's global allocator; enabling both is rejected at compile time.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; the implementation lives in `selection.rs`.

mod selection;

#[cfg(test)]
mod allocator_tests;

pub use selection::{SKAllocatorKind, sk_allocator_kind};
