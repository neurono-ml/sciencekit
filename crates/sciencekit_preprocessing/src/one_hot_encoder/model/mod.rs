//! Pure dispatcher: the fitted one-hot encoder model and its transforms.
//! Carries no implementation logic.

mod core_implementation;
mod sparse;

pub use core_implementation::SKOneHotEncoderModel;
