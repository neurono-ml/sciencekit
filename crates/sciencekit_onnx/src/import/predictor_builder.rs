//! Builder for the imported-model predictor.

use std::marker::PhantomData;
use std::path::PathBuf;

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::SKOnnxImportError;
use super::interface::interface;
use super::predictor::SKOnnxPredictor;
use crate::graph::SKOnnxModel;

/// Builder for [`SKOnnxPredictor`].
#[derive(Debug, Clone)]
pub struct SKOnnxPredictorBuilder<F: SKFloat> {
    model: Option<SKOnnxModel>,
    bytes: Option<Vec<u8>>,
    path: Option<PathBuf>,
    backend: SKBackendKind,
    execution_mode: SKExecutionMode,
    marker: PhantomData<F>,
}

impl<F: SKFloat> Default for SKOnnxPredictorBuilder<F> {
    fn default() -> Self {
        SKOnnxPredictorBuilder::new()
    }
}

impl<F: SKFloat> SKOnnxPredictorBuilder<F> {
    /// A builder with no model attached.
    pub fn new() -> Self {
        SKOnnxPredictorBuilder {
            model: None,
            bytes: None,
            path: None,
            backend: SKBackendKind::Faer,
            execution_mode: SKExecutionMode::Automatic,
            marker: PhantomData,
        }
    }

    /// Set the math backend kind recorded in every prediction span
    /// (default [`SKBackendKind::Faer`]).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Set the execution mode recorded in every prediction span
    /// (default [`SKExecutionMode::Automatic`]).
    pub fn execution_mode(mut self, execution_mode: SKExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    /// Attach an already decoded model.
    pub fn model(mut self, model: SKOnnxModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Attach the model's protobuf bytes.
    pub fn model_bytes(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.bytes = Some(bytes.into());
        self
    }

    /// Attach a model file path.
    pub fn model_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Validate the model and produce the predictor.
    pub fn build(self) -> Result<SKOnnxPredictor<F>, SKOnnxImportError> {
        let model = match (self.model, self.bytes, self.path) {
            (Some(model), _, _) => model,
            (None, Some(bytes), _) => SKOnnxModel::from_bytes(bytes)?,
            (None, None, Some(path)) => SKOnnxModel::from_file(path)?,
            (None, None, None) => {
                return Err(SKOnnxImportError::InvalidModel {
                    detail: "no model bytes, model or file was supplied".to_string(),
                });
            }
        };
        let interface = interface(&model)?;
        Ok(SKOnnxPredictor {
            model,
            interface,
            backend: self.backend,
            execution_mode: self.execution_mode,
            marker: PhantomData,
        })
    }
}
