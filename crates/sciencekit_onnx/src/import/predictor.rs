//! The generic predictor over an imported ONNX model.

use std::collections::BTreeMap;

use std::collections::BTreeMap as ArrayMap;

use ndarray::{Array1, ArrayD, IxDyn};
use sciencekit_common::{
    SKBackendKind, SKDataView, SKExecutionMode, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use super::SKOnnxImportError;
use super::interface::{SKOnnxInterface, SKOnnxPredictionHead};
use super::interpreter;
use super::predictor_builder::SKOnnxPredictorBuilder;
use super::value::SKOnnxValue;
use crate::graph::SKOnnxModel;

/// Runs an imported ONNX graph behind the library's prediction traits.
///
/// The same predictor implements both trait surfaces; a method whose head the
/// model does not declare returns [`sciencekit_common::SKError::UnsupportedOperation`]
/// instead of guessing.
#[derive(Debug, Clone)]
pub struct SKOnnxPredictor<F: SKFloat> {
    pub(super) model: SKOnnxModel,
    pub(super) interface: SKOnnxInterface,
    pub(super) backend: SKBackendKind,
    pub(super) execution_mode: SKExecutionMode,
    pub(super) marker: std::marker::PhantomData<F>,
}

impl<F: SKFloat> SKOnnxPredictor<F> {
    /// Start configuring a predictor.
    pub fn builder() -> SKOnnxPredictorBuilder<F> {
        SKOnnxPredictorBuilder::new()
    }

    /// The declared feature width the input must match.
    pub fn feature_count(&self) -> usize {
        self.interface.feature_count
    }

    /// The math backend kind recorded in every prediction span.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution mode recorded in every prediction span.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }

    /// The loaded model.
    pub fn model(&self) -> &SKOnnxModel {
        &self.model
    }

    /// Run the graph and return every declared output as an array.
    ///
    /// This is the generic surface behind the typed traits: keys are the ONNX
    /// output names exactly as the model declares them.
    pub fn predict_arrays<'a, X>(
        &self,
        features: X,
    ) -> Result<ArrayMap<String, ArrayD<F>>, SKOnnxImportError>
    where
        X: TryInto<SKDataView<'a, F>, Error = sciencekit_common::SKError>,
    {
        self.run_observed(features, |rows, values| {
            let outputs = self.run(rows, values)?;
            outputs
                .into_iter()
                .map(|(name, value)| {
                    let shape = value.shape();
                    let converted: Vec<F> = value
                        .to_f64_values()
                        .into_iter()
                        .map(|value| num_traits::NumCast::from(value).unwrap_or_else(F::zero))
                        .collect();
                    let array = ArrayD::from_shape_vec(IxDyn(&shape), converted).map_err(|_| {
                        SKOnnxImportError::ShapeMismatch {
                            expected: shape.clone(),
                            found: shape,
                        }
                    })?;
                    Ok((name, array))
                })
                .collect()
        })
    }

    /// Run one prediction inside its observability span.
    ///
    /// The span records the configured backend and execution mode - never a
    /// hardcoded value - so a consumer's subscriber sees the selection the
    /// builder received.
    pub(super) fn run_observed<'a, X, T>(
        &self,
        features: X,
        operation: impl FnOnce(usize, Vec<f64>) -> Result<T, SKOnnxImportError>,
    ) -> Result<T, SKOnnxImportError>
    where
        X: TryInto<SKDataView<'a, F>, Error = sciencekit_common::SKError>,
    {
        let view = features.try_into().map_err(SKOnnxImportError::Central)?;
        let dense = view.as_dense().map_err(SKOnnxImportError::Central)?;
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count: dense.nrows(),
            columns_count: dense.ncols(),
            execution_mode: self.execution_mode,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let (rows, values) = self.features(dense)?;
            operation(rows, values)
        })
    }

    pub(super) fn features(
        &self,
        dense: ndarray::ArrayView2<'_, F>,
    ) -> Result<(usize, Vec<f64>), SKOnnxImportError> {
        if dense.ncols() != self.interface.feature_count {
            return Err(SKOnnxImportError::ShapeMismatch {
                expected: vec![dense.nrows(), self.interface.feature_count],
                found: vec![dense.nrows(), dense.ncols()],
            });
        }
        let mut values = Vec::with_capacity(dense.len());
        for (position, value) in dense.iter().enumerate() {
            let value = value.to_f64().unwrap_or(f64::NAN);
            if !value.is_finite() {
                return Err(SKOnnxImportError::Central(
                    sciencekit_common::SKError::NonFiniteValue { position },
                ));
            }
            values.push(value);
        }
        Ok((dense.nrows(), values))
    }

    pub(super) fn run(
        &self,
        rows: usize,
        values: Vec<f64>,
    ) -> Result<BTreeMap<String, SKOnnxValue>, SKOnnxImportError> {
        let input = SKOnnxValue::from_f64_values(
            self.interface.input_element_type,
            vec![rows, self.interface.feature_count],
            values,
        )?;
        let mut inputs = BTreeMap::new();
        inputs.insert(self.interface.input_name.clone(), input);
        let names = output_names(&self.interface.head);
        interpreter::execute(
            &self.model.model().graph.initializers,
            &self.model.model().graph.nodes,
            &inputs,
            &names,
        )
    }

    pub(super) fn predict_float_rows(
        &self,
        rows: usize,
        values: Vec<f64>,
    ) -> Result<Array1<F>, SKOnnxImportError> {
        let outputs = self.run(rows, values)?;
        let (name, width) = match &self.interface.head {
            SKOnnxPredictionHead::Regression { output, width, .. } => (output, *width),
            _ => {
                return Err(SKOnnxImportError::Central(
                    sciencekit_common::SKError::UnsupportedOperation {
                        operation: "onnx_regression",
                        reason: "the imported model does not declare a regression head".to_string(),
                    },
                ));
            }
        };
        let value = outputs
            .get(name)
            .ok_or_else(|| SKOnnxImportError::UnsupportedOutput {
                detail: format!("the regression output `{name}` was not produced"),
            })?;
        if width != 1 {
            return Err(SKOnnxImportError::Central(
                sciencekit_common::SKError::UnsupportedOperation {
                    operation: "onnx_regression",
                    reason: format!(
                        "the regression output has width {width}; predict returns one response per row, use predict_arrays for multi-column outputs"
                    ),
                },
            ));
        }
        Ok(Array1::from_iter(value.to_f64_values().into_iter().map(
            |value| num_traits::NumCast::from(value).unwrap_or_else(F::zero),
        )))
    }
}

/// The graph outputs the active head needs.
pub(super) fn output_names(head: &SKOnnxPredictionHead) -> Vec<String> {
    match head {
        SKOnnxPredictionHead::Regression { output, .. } => vec![output.clone()],
        SKOnnxPredictionHead::Classification {
            labels,
            probabilities,
            ..
        } => vec![labels.clone(), probabilities.clone()],
        SKOnnxPredictionHead::LabelsOnly { labels } => vec![labels.clone()],
    }
}
