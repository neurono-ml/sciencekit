//! The fitted imputer: per-column statistics and dropped-column bookkeeping
//! (spec `simple-imputer`).

mod core_implementation;
mod partial_fit;
mod transform;

pub use core_implementation::SKSimpleImputerModel;
