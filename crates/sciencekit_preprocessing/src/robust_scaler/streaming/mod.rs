//! Pure dispatcher: the robust scaler streaming submodules. Carries no
//! implementation logic.

pub(super) mod batches;
pub(super) mod collected_sparse;
mod fit;
pub(super) mod state;
