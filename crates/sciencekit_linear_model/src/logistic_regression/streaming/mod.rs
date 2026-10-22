//! Pure dispatcher: the out-of-core logistic fit. Carries no implementation
//! logic.

mod fit;

pub(crate) use fit::sk_fit_streaming;
