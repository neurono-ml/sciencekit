//! Test-only mock models exercising the export/import contracts.

use ndarray::ArrayView1;

use crate::SKError;
use crate::sk_float::SKFloat;

use super::tensor::tensor_dtype_for_scalar;
use super::{
    SKExportable, SKImportable, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    SKTensorValues,
};

/// The estimator name every mock export carries.
pub(crate) const MOCK_ESTIMATOR_NAME: &str = "SKMockLinearModel";
/// The format version every mock export writes.
pub(crate) const MOCK_FORMAT_VERSION: u32 = 1;

/// A fitted linear model standing in for a real estimator in contract tests.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SKMockLinearModel<F: SKFloat> {
    coefficients: Vec<F>,
    intercept: F,
    class_labels: Option<Vec<String>>,
}

/// Build the canonical three-feature mock used across the interop tests.
pub(crate) fn mock_linear_model() -> SKMockLinearModel<f64> {
    SKMockLinearModel {
        coefficients: vec![0.5, -1.5, 2.0],
        intercept: 0.25,
        class_labels: None,
    }
}

impl<F: SKFloat> SKMockLinearModel<F> {
    /// The number of features the model consumes.
    pub(crate) fn feature_count(&self) -> usize {
        self.coefficients.len()
    }

    /// Attach canonical class labels, emulating a classifier.
    pub(crate) fn with_class_labels(mut self, labels: Vec<String>) -> Self {
        self.class_labels = Some(labels);
        self
    }

    /// Score one row as `x·coefficients + intercept`.
    pub(crate) fn score(&self, features: ArrayView1<'_, F>) -> F {
        let inner = features
            .iter()
            .zip(self.coefficients.iter())
            .map(|(feature, coefficient)| *feature * *coefficient)
            .fold(F::zero(), |sum, term| sum + term);
        inner + self.intercept
    }
}

impl<F: SKFloat> SKExportable<F> for SKMockLinearModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = SKModelMetadata::new(
            MOCK_ESTIMATOR_NAME,
            MOCK_FORMAT_VERSION,
            tensor_dtype_for_scalar::<F>(),
            self.feature_count(),
        )?
        .with_parameter("intercept", SKParameterValue::from(true))?
        .with_parameter("solver", SKParameterValue::from("mock"))?;
        if let Some(labels) = &self.class_labels {
            metadata = metadata.with_class_labels(labels.clone())?;
        }
        let coefficients = SKTensor::floating(
            "linear.coefficients",
            vec![self.feature_count()],
            self.coefficients.clone(),
        )?;
        let intercept = SKTensor::floating("linear.intercept", Vec::new(), vec![self.intercept])?;
        SKModelState::new(metadata, vec![coefficients, intercept])
    }
}

impl<F: SKFloat> SKImportable<F> for SKMockLinearModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let coefficients = floating_values(state, "linear.coefficients")?;
        let intercept = floating_values(state, "linear.intercept")?;
        if intercept.len() != 1 {
            return Err(SKError::InvalidHyperparameter {
                name: "linear.intercept",
                reason: "the intercept tensor must hold exactly one value".to_string(),
            });
        }
        super::sk_validate_import_metadata(
            state,
            MOCK_ESTIMATOR_NAME,
            MOCK_FORMAT_VERSION,
            coefficients.len(),
        )?;
        Ok(SKMockLinearModel {
            coefficients,
            intercept: intercept[0],
            class_labels: state.metadata().class_labels().map(<[String]>::to_vec),
        })
    }
}

/// Read a floating tensor's values or report a malformed state.
fn floating_values<F: SKFloat>(
    state: &SKModelState<F>,
    name: &'static str,
) -> Result<Vec<F>, SKError> {
    let tensor = state.tensor(name).ok_or(SKError::InvalidHyperparameter {
        name,
        reason: "the exported state does not carry this tensor".to_string(),
    })?;
    match tensor.values() {
        SKTensorValues::Floating(values) => Ok(values.clone()),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "non-floating tensor",
            suggestion: "export this tensor with floating values",
        }),
    }
}

/// A model that deliberately refuses reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SKUnsupportedMockModel;

impl<F: SKFloat> SKImportable<F> for SKUnsupportedMockModel {
    fn import_state(_state: &SKModelState<F>) -> Result<Self, SKError> {
        Err(SKError::UnsupportedOperation {
            operation: "import_state",
            reason: "SKUnsupportedMockModel never reconstructs learned state".to_string(),
        })
    }
}
