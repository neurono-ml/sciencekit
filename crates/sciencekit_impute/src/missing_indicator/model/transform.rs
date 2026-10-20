//! The mask transform: dense marking and the CSR mask (spec
//! `missing-indicator`).
//!
//! Not implemented yet: the red contract of task 8.1.

use sciencekit_common::{SKDataView, SKError, SKFeatureTransformer, SKFloat};

use crate::missing_indicator::model::SKMissingIndicatorModel;
use crate::missing_indicator::output::SKIndicatorOutput;

impl<F: SKFloat> SKFeatureTransformer<F> for SKMissingIndicatorModel<F> {
    type Output = SKIndicatorOutput<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, _features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "the mask transform is not implemented yet",
        })
    }
}
