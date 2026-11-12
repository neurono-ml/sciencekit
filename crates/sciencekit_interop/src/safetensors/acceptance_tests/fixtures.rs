//! Acceptance fixtures, span recorder and dense/CSR linear regressor
//! (PRD §8.7, spec `safetensors-format`).

use std::sync::{Arc, Mutex};

use ndarray::{Array1, Axis};
use sciencekit_common::{
    SKDataView, SKError, SKExportable, SKImportable, SKModelMetadata, SKModelState,
    SKParameterValue, SKRegressorPredictor, SKTensor, SKTensorDataType, SKTensorValues,
    sk_validate_import_metadata,
};
use tracing::Subscriber;
use tracing::subscriber::Interest;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

#[derive(Default, Clone)]
pub(super) struct SpanRecorder {
    pub(super) names: Arc<Mutex<Vec<String>>>,
}

impl<S: Subscriber> Layer<S> for SpanRecorder {
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(
        &self,
        attributes: &tracing::span::Attributes<'_>,
        _identifier: &tracing::span::Id,
        _context: Context<'_, S>,
    ) {
        self.names
            .lock()
            .unwrap()
            .push(attributes.metadata().name().to_string());
    }
}

#[derive(Debug, PartialEq)]
pub(super) struct AcceptanceLinearRegressor {
    pub(super) coefficients: Vec<f64>,
    pub(super) intercept: f64,
}

impl SKExportable<f64> for AcceptanceLinearRegressor {
    fn export_state(&self) -> Result<SKModelState<f64>, SKError> {
        let metadata = SKModelMetadata::new(
            "SKLinearRegression",
            1,
            SKTensorDataType::F64,
            self.coefficients.len(),
        )?
        .with_parameter("include_intercept", SKParameterValue::from(true))?;
        let weights = SKTensor::floating(
            "linear.coefficients",
            vec![self.coefficients.len()],
            self.coefficients.clone(),
        )?;
        let bias = SKTensor::floating("linear.intercept", vec![], vec![self.intercept])?;
        SKModelState::new(metadata, vec![weights, bias])
    }
}

impl SKImportable<f64> for AcceptanceLinearRegressor {
    fn import_state(state: &SKModelState<f64>) -> Result<Self, SKError> {
        sk_validate_import_metadata(
            state,
            "SKLinearRegression",
            1,
            state.metadata().feature_count(),
        )?;
        let weights = match state.tensor("linear.coefficients").map(SKTensor::values) {
            Some(SKTensorValues::Floating(values)) => values.clone(),
            _ => {
                return Err(SKError::InvalidHyperparameter {
                    name: "linear.coefficients",
                    reason: "missing floating coefficients".to_string(),
                });
            }
        };
        let intercept = match state.tensor("linear.intercept").map(SKTensor::values) {
            Some(SKTensorValues::Floating(values)) if values.len() == 1 => values[0],
            _ => {
                return Err(SKError::InvalidHyperparameter {
                    name: "linear.intercept",
                    reason: "missing scalar intercept".to_string(),
                });
            }
        };
        Ok(Self {
            coefficients: weights,
            intercept,
        })
    }
}

impl SKRegressorPredictor<f64> for AcceptanceLinearRegressor {
    type Error = SKError;

    fn predict<'view, X>(&self, features: X) -> Result<Array1<f64>, SKError>
    where
        X: TryInto<SKDataView<'view, f64>, Error = SKError>,
    {
        match features.try_into()? {
            SKDataView::Dense(matrix) => Ok(matrix.map_axis(Axis(1), |row| {
                row.iter()
                    .zip(&self.coefficients)
                    .map(|(feature, weight)| feature * weight)
                    .sum::<f64>()
                    + self.intercept
            })),
            SKDataView::Sparse(csr) if csr.is_csr() => {
                let predictions: Vec<f64> = csr
                    .outer_iterator()
                    .map(|row| {
                        row.iter()
                            .map(|(column, &value)| value * self.coefficients[column])
                            .sum::<f64>()
                            + self.intercept
                    })
                    .collect();
                Ok(Array1::from_vec(predictions))
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "non-CSR sparse matrix",
                suggestion: "pass a dense ArrayView2 or borrowed CSR view",
            }),
        }
    }
}
