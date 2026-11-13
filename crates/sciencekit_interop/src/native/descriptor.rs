//! Descriptive metadata carried beside a reconstructed native model
//! (spec `native-model-import`, requirement "Safetensors native import").

use sciencekit_common::{SKBackendKind, SKFloat, SKModelState, sk_parse_backend_parameter};

/// Whether the imported configuration can be continued with more data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKRetrainingSupport {
    /// The model keeps the learning state needed for `partial_fit`.
    Supported,
    /// The model is inference-only after conversion.
    Unsupported {
        /// Why continued fitting is unavailable.
        reason: &'static str,
    },
}

impl SKRetrainingSupport {
    /// Whether continued fitting is available.
    pub fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }

    /// The guidance shown when continued fitting is requested and unavailable.
    pub fn reason(self) -> Option<&'static str> {
        match self {
            Self::Supported => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// What a native conversion produced, independent of the model family.
///
/// The descriptor preserves exactly the metadata a caller needs to keep using
/// the model: its stable estimator name, the feature width it was fitted on, the
/// canonical class table for a classifier and the backend kind it records in its
/// observability spans.
#[derive(Debug, Clone, PartialEq)]
pub struct SKNativeModelDescriptor {
    estimator_name: String,
    feature_count: usize,
    class_labels: Option<Vec<String>>,
    backend: SKBackendKind,
    retraining: SKRetrainingSupport,
}

impl SKNativeModelDescriptor {
    /// Build a descriptor from the imported state and the family's retraining rule.
    pub fn from_state<F: SKFloat>(
        state: &SKModelState<F>,
        retraining: SKRetrainingSupport,
    ) -> Self {
        let metadata = state.metadata();
        Self {
            estimator_name: metadata.estimator_name().to_string(),
            feature_count: metadata.feature_count(),
            class_labels: metadata.class_labels().map(<[String]>::to_vec),
            backend: sk_parse_backend_parameter(metadata),
            retraining,
        }
    }

    /// The stable estimator name the source state declared.
    pub fn estimator_name(&self) -> &str {
        &self.estimator_name
    }

    /// The feature width the model was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// The canonical class labels in index order, for a classifier.
    pub fn class_labels(&self) -> Option<&[String]> {
        self.class_labels.as_deref()
    }

    /// The backend kind the model records in every operation span.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Whether continued fitting is available.
    pub fn supports_retraining(&self) -> bool {
        self.retraining.is_supported()
    }

    /// The retraining rule carried by this descriptor.
    pub fn retraining(&self) -> SKRetrainingSupport {
        self.retraining
    }
}
