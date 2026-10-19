pub mod builder;
pub mod core_implementation;
pub mod model;
pub mod streaming;

#[cfg(test)]
mod minmax_scaler_tests;
#[cfg(test)]
mod partial_fit_tests;
#[cfg(test)]
mod streaming_tests;
#[cfg(test)]
mod test_datasets;
