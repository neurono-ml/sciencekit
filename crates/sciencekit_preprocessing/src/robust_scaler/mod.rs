pub mod builder;
pub(crate) mod configuration;
pub mod core_implementation;
pub mod model;
pub mod quantiles;
pub(crate) mod sketch;
pub mod streaming;

#[cfg(test)]
mod partial_fit_tests;
#[cfg(test)]
mod quantiles_tests;
#[cfg(test)]
mod robust_scaler_tests;
#[cfg(test)]
mod streaming_tests;
#[cfg(test)]
mod test_datasets;
