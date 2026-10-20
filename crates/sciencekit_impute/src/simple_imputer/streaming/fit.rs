//! Out-of-core fit over a lazy source (spec `simple-imputer`).
//!
//! Not implemented yet: the red contract of task 7.1.

use sciencekit_common::{SKError, SKFloat, SKLazySource};

use super::super::core_implementation::SKSimpleImputer;
use super::super::model::SKSimpleImputerModel;

impl<F: SKFloat> SKSimpleImputer<F> {
    /// Fit on a sequential batch source.
    pub fn fit_streaming<S>(&self, _source: &mut S) -> Result<SKSimpleImputerModel<F>, SKError>
    where
        S: SKLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        Err(SKError::Conversion("streaming fit is not implemented".into()))
    }
}
