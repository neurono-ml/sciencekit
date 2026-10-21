//! Pure dispatcher: the shared fitted-model core of the L1-regularized
//! estimators. Carries no implementation logic.

mod accessors;
mod definition;
mod incremental;
mod prediction;

pub(crate) use definition::SKCoordinateDescentModelCore;
