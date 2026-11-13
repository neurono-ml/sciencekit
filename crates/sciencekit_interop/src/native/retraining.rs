//! Continued fitting on an imported native model
//! (spec `native-model-import`, requirement "Continued training").
//!
//! Only families that keep the learning state a `partial_fit` call needs accept
//! more data. Everything else fails with `UnsupportedOperation` and the guidance
//! recorded in its descriptor — never with a silent reset of the imported
//! coefficients.

use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat, SKTargetView};
use sciencekit_math::SKBackendSelection;

use super::conversion::SKNativeImport;
use super::model::{SKNativeClassifier, SKNativeRegressor};

impl<F: SKFloat + SKBackendSelection> SKNativeImport<F> {
    /// Continue fitting from the imported coefficients and the retained state.
    ///
    /// Returns the model itself for chaining. A family whose imported state is
    /// inference-only returns `UnsupportedOperation` with its documented
    /// guidance; a wrong feature width or an unseen class is rejected by the
    /// underlying model exactly as in a fresh `partial_fit`.
    pub fn partial_fit<'a>(
        &mut self,
        features: ArrayView2<'a, F>,
        targets: SKTargetView<'a, F>,
    ) -> Result<&mut Self, SKError> {
        let continuous = targets.as_continuous()?;
        let target_view = continuous.view();
        match self {
            Self::Regressor { model, descriptor } => {
                match model {
                    SKNativeRegressor::LinearRegression(model) => {
                        model.partial_fit(features, target_view)?;
                    }
                    SKNativeRegressor::StochasticGradient(model) => {
                        model.partial_fit(features, target_view)?;
                    }
                    _ => return Err(inference_only(descriptor)),
                }
                Ok(self)
            }
            Self::Classifier { model, descriptor } => {
                match model {
                    SKNativeClassifier::LogisticRegression(model) => {
                        model.partial_fit(features, target_view)?;
                    }
                    SKNativeClassifier::StochasticGradient(model) => {
                        model.partial_fit(features, target_view)?;
                    }
                    _ => return Err(inference_only(descriptor)),
                }
                Ok(self)
            }
            _ => Err(inference_only(self.descriptor())),
        }
    }
}

fn inference_only(descriptor: &super::descriptor::SKNativeModelDescriptor) -> SKError {
    let reason = descriptor
        .retraining()
        .reason()
        .unwrap_or("this family does not retain a learning state")
        .to_string();
    SKError::UnsupportedOperation {
        operation: "partial_fit",
        reason: format!(
            "the imported {} cannot continue training: {reason}",
            descriptor.estimator_name()
        ),
    }
}
