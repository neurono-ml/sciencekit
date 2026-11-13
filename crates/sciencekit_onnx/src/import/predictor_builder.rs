//! Builder for the imported-model predictor.

use std::marker::PhantomData;
use std::path::PathBuf;

use sciencekit_common::SKFloat;

use super::SKOnnxImportError;
use super::interface::interface;
use super::predictor::SKOnnxPredictor;
use crate::graph::SKOnnxModel;

/// Builder for [`SKOnnxPredictor`].
#[derive(Debug, Clone, Default)]
pub struct SKOnnxPredictorBuilder<F: SKFloat> {
    model: Option<SKOnnxModel>,
    bytes: Option<Vec<u8>>,
    path: Option<PathBuf>,
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKOnnxPredictorBuilder<F> {
    /// A builder with no model attached.
    pub fn new() -> Self {
        SKOnnxPredictorBuilder {
            model: None,
            bytes: None,
            path: None,
            marker: PhantomData,
        }
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
            marker: PhantomData,
        })
    }
}
