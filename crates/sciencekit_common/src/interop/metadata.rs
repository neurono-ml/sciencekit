//! Estimator metadata (spec `model-export-contracts`).

use std::collections::{BTreeMap, BTreeSet};

use crate::SKError;

use super::{SKParameterValue, SKTensorDType};

/// The descriptive part of an exported model state.
///
/// Metadata names the estimator, its format version, the scalar dtype and the
/// feature count, and carries class labels when applicable plus a typed
/// hyperparameter map. Construction validates identity and label uniqueness.
#[derive(Debug, Clone, PartialEq)]
pub struct SKModelMetadata {
    estimator_name: String,
    format_version: u32,
    scalar_dtype: SKTensorDType,
    feature_count: usize,
    class_labels: Option<Vec<String>>,
    parameters: BTreeMap<String, SKParameterValue>,
}

impl SKModelMetadata {
    /// Build metadata for a model, rejecting an empty estimator name.
    pub fn new(
        estimator_name: impl Into<String>,
        format_version: u32,
        scalar_dtype: SKTensorDType,
        feature_count: usize,
    ) -> Result<Self, SKError> {
        let estimator_name = estimator_name.into();
        if estimator_name.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "estimator_name",
                reason: "an estimator name must not be empty".to_string(),
            });
        }
        Ok(SKModelMetadata {
            estimator_name,
            format_version,
            scalar_dtype,
            feature_count,
            class_labels: None,
            parameters: BTreeMap::new(),
        })
    }

    /// Attach the canonical class labels, rejecting an empty or duplicated set.
    ///
    /// The labels are stored in index order, so the canonical table stays
    /// reversible after a round-trip.
    pub fn with_class_labels(mut self, labels: Vec<String>) -> Result<Self, SKError> {
        if labels.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "class labels must not be empty; omit them for a regressor".to_string(),
            });
        }
        let distinct: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
        if distinct.len() != labels.len() {
            return Err(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "class labels must be unique".to_string(),
            });
        }
        self.class_labels = Some(labels);
        Ok(self)
    }

    /// Attach a typed hyperparameter under a non-empty name.
    ///
    /// Re-using a name replaces its previous value, mirroring map semantics.
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: SKParameterValue,
    ) -> Result<Self, SKError> {
        let name = name.into();
        if name.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "parameter_name",
                reason: "a hyperparameter name must not be empty".to_string(),
            });
        }
        self.parameters.insert(name, value);
        Ok(self)
    }

    /// The estimator's stable name.
    pub fn estimator_name(&self) -> &str {
        &self.estimator_name
    }

    /// The format version the exporting model wrote.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// The model scalar dtype.
    pub fn scalar_dtype(&self) -> SKTensorDType {
        self.scalar_dtype
    }

    /// The number of features the model was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// The canonical class labels in index order, when the model is a classifier.
    pub fn class_labels(&self) -> Option<&[String]> {
        self.class_labels.as_deref()
    }

    /// The typed hyperparameter map.
    pub fn parameters(&self) -> &BTreeMap<String, SKParameterValue> {
        &self.parameters
    }

    /// A single hyperparameter by name.
    pub fn parameter(&self, name: &str) -> Option<&SKParameterValue> {
        self.parameters.get(name)
    }
}
