//! Builder for the ONNX native importer (spec `native-model-import`).

use std::path::PathBuf;
use std::time::Instant;

use sciencekit_common::{
    SKBackendKind, SKExecutionMode, SKFloat, SKModelState, SKParameterValue, sk_backend_parameter,
};
use sciencekit_interop::native::{SKNativeImport, sk_import_native_state};

use super::error::SKOnnxNativeError;
use super::state::sk_state_from_onnx;
use crate::graph::SKOnnxModel;

/// Converts an ONNX graph into a native sciencekit model.
///
/// Construct exclusively via [`SKOnnxNativeImporter::builder`]. The generic
/// scalar `F` is chosen at [`SKOnnxNativeImporter::build`], so the builder itself
/// is not generic.
#[derive(Debug, Clone)]
pub struct SKOnnxNativeImporter {
    model: Option<SKOnnxModel>,
    bytes: Option<Vec<u8>>,
    path: Option<PathBuf>,
    backend: SKBackendKind,
    execution_mode: SKExecutionMode,
}

impl Default for SKOnnxNativeImporter {
    fn default() -> Self {
        Self::builder()
    }
}

impl SKOnnxNativeImporter {
    /// Start configuring an importer with the default backend and execution mode.
    pub fn builder() -> Self {
        Self {
            model: None,
            bytes: None,
            path: None,
            backend: SKBackendKind::Faer,
            execution_mode: SKExecutionMode::Automatic,
        }
    }

    /// Set the math backend kind recorded by the converted model (default `Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Set the execution mode recorded by the converted model.
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

    /// Decode the graph, map it to a neutral state and build the native model.
    pub fn build<F: SKFloat>(self) -> Result<SKNativeImport<F>, SKOnnxNativeError> {
        let span = tracing::info_span!(
            "sciencekit.onnx.native_import",
            estimator = tracing::field::Empty,
            feature_count = tracing::field::Empty,
            backend = %self.backend,
            execution_mode = ?self.execution_mode,
            duration_ms = tracing::field::Empty,
            error = tracing::field::Empty,
        );
        let started_at = Instant::now();
        let result = span.in_scope(|| self.convert::<F>());
        match &result {
            Ok(imported) => {
                span.record("estimator", imported.descriptor().estimator_name());
                span.record("feature_count", imported.descriptor().feature_count());
            }
            Err(error) => {
                span.record("error", error.to_string());
            }
        }
        span.record("duration_ms", started_at.elapsed().as_millis() as u64);
        result
    }

    fn convert<F: SKFloat>(self) -> Result<SKNativeImport<F>, SKOnnxNativeError> {
        let model = match (self.model, self.bytes, self.path) {
            (Some(model), _, _) => model,
            (None, Some(bytes), _) => SKOnnxModel::from_bytes(bytes)?,
            (None, None, Some(path)) => SKOnnxModel::from_file(path)?,
            (None, None, None) => {
                return Err(SKOnnxNativeError::InvalidGraph {
                    detail: "no model bytes, model or file was supplied".to_string(),
                });
            }
        };
        let state = sk_state_from_onnx::<F>(&model)?;
        let state = with_backend(state, self.backend)?;
        Ok(sk_import_native_state(&state)?)
    }
}

/// Record the requested backend kind in the state metadata the model imports.
fn with_backend<F: SKFloat>(
    state: SKModelState<F>,
    backend: SKBackendKind,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let metadata = state
        .metadata()
        .clone()
        .with_parameter("backend", sk_backend_parameter(backend))
        .map_err(native_error)?;
    let tensors = state.tensors().values().cloned().collect();
    SKModelState::new(metadata, tensors).map_err(native_error)
}

fn native_error(error: sciencekit_common::SKError) -> SKOnnxNativeError {
    SKOnnxNativeError::Native(sciencekit_interop::native::SKNativeImportError::from(error))
}

/// A parameter value carrying the requested backend, re-exported for tests.
#[allow(dead_code)]
fn backend_parameter(backend: SKBackendKind) -> SKParameterValue {
    sk_backend_parameter(backend)
}
