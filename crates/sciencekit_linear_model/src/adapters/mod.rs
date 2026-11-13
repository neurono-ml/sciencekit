//! Model adapters that fit a small correction on top of a frozen fitted model.
//!
//! The module is a **pure dispatcher**: it declares and re-exports the adapter
//! families. Low-rank adaptation lives under [`lora`].

pub mod lora;
