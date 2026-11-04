//! The classifier's multi-epoch stream.

mod fit;
mod state;

pub(crate) use fit::sk_fit_streaming;
