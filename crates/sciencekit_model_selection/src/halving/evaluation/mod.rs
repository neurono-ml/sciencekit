//! Pure dispatcher for successive-halving round evaluation.

mod runner;
mod validation;

pub(crate) use runner::{SKHalvingEvaluationRequest, sk_run_halving_search};
