//! The out-of-core fit (spec `simple-imputer`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod batches;
mod fit;
mod state;
