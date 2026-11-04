//! Pure dispatcher: the out-of-core logistic fit. Carries no implementation
//! logic.

mod accumulator;
mod fit;

pub(crate) use fit::sk_fit_streaming;
