//! Continued fitting on an imported native model
//! (spec `native-model-import`, requirement "Continued training").
//!
//! Only families that keep the learning state a `partial_fit` call needs accept
//! more data. Everything else fails with `UnsupportedOperation` and the guidance
//! recorded in its descriptor — never with a silent reset of the imported
//! coefficients.

use ndarray::ArrayView2;
use sciencekit_common::{SKBuilder, SKError, SKFloat, SKSupervisedFit, SKTargetView};
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
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
        match self {
            Self::Regressor { model, descriptor } => {
                match model {
                    SKNativeRegressor::StochasticGradient(model) => match targets {
                        SKTargetView::Continuous(view) => {
                            model.partial_fit(features, view)?;
                        }
                        SKTargetView::Integer(view) => {
                            model.partial_fit(features, view)?;
                        }
                        SKTargetView::Nominal(labels) => {
                            model.partial_fit(features, labels)?;
                        }
                        _ => return Err(unsupported_target()),
                    },
                    _ => return Err(inference_only(descriptor)),
                }
                Ok(self)
            }
            Self::Classifier { model, descriptor } => {
                match model {
                    SKNativeClassifier::LogisticRegression(current) => {
                        let builder = SKLogisticRegression::<F>::build()
                            .penalty(current.penalty())
                            .solver(current.solver())
                            .alpha(current.alpha())
                            .l1_ratio(current.l1_ratio())
                            .class_weight(current.class_weight().clone())
                            .include_intercept(current.include_intercept())
                            .random_seed(current.random_seed())
                            .warm_start_from(current)
                            .build()?;
                        *current = match targets {
                            SKTargetView::Continuous(_) => {
                                return Err(SKError::UnsupportedRepresentation {
                                    representation: "continuous",
                                    suggestion: "pass integer or nominal class labels to a classifier",
                                });
                            }
                            SKTargetView::Integer(view) => builder.fit(features, view)?,
                            SKTargetView::Nominal(labels) => builder.fit(features, labels)?,
                            _ => return Err(unsupported_target()),
                        };
                    }
                    SKNativeClassifier::StochasticGradient(model) => match targets {
                        SKTargetView::Continuous(view) => {
                            model.partial_fit(features, view)?;
                        }
                        SKTargetView::Integer(view) => {
                            model.partial_fit(features, view)?;
                        }
                        SKTargetView::Nominal(labels) => {
                            model.partial_fit(features, labels)?;
                        }
                        _ => return Err(unsupported_target()),
                    },
                    _ => return Err(inference_only(descriptor)),
                }
                Ok(self)
            }
            _ => Err(inference_only(self.descriptor())),
        }
    }
}

fn unsupported_target() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "target",
        suggestion: "pass continuous values to a regressor or class labels to a classifier",
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
