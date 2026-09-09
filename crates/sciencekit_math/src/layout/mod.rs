//! Memory-layout helpers (spec `layout`, PRD §4.3).
//!
//! Hot kernels want contiguous inputs. These helpers detect the layout of an
//! [`ArrayView`] and produce a contiguous owner only when needed, so calling
//! code can feed zero-copy views into SIMD loops without surprises.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; the implementation lives in `memory_layout.rs`.

mod memory_layout;

pub use memory_layout::{
    SKMemoryLayout, sk_force_contiguous, sk_is_c_contiguous, sk_is_f_contiguous, sk_memory_layout,
};

#[cfg(test)]
mod layout_tests;
