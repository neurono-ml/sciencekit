//! The fitted imputer: per-column statistics and dropped-column bookkeeping
//! (spec `simple-imputer`).

mod core_implementation;
mod transform;

pub use core_implementation::SKSimpleImputerModel;
