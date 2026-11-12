//! Column selection and conversion options for a tabular source (spec
//! `tabular-sources`).

use sciencekit_common::SKError;

use super::SKColumnKind;
use crate::batch::SKFeatureLayout;
use crate::missing::SKMissingValuePolicy;

/// The columns a tabular source keeps and how each one enters the dataset.
///
/// Construction validates the surface: at least one column must be selected and
/// no name may appear twice. Resolution against the actual source columns
/// happens in the adapter, before any row is read.
#[derive(Debug, Clone, PartialEq)]
pub struct SKTabularSourceConfig {
    columns: Vec<(String, SKColumnKind)>,
    policy: SKMissingValuePolicy,
    layout: SKFeatureLayout,
}

impl SKTabularSourceConfig {
    /// Start building a configuration.
    pub fn builder() -> SKTabularSourceConfigBuilder {
        SKTabularSourceConfigBuilder::default()
    }

    /// The configured columns in declaration order.
    pub fn columns(&self) -> &[(String, SKColumnKind)] {
        &self.columns
    }

    /// The kind declared for a column name.
    pub fn kind_of(&self, name: &str) -> Option<SKColumnKind> {
        self.columns
            .iter()
            .find(|(existing, _)| existing == name)
            .map(|(_, kind)| *kind)
    }

    /// The missing-value policy applied to every null.
    pub fn missing_value_policy(&self) -> SKMissingValuePolicy {
        self.policy
    }

    /// The representation requested for the feature block.
    pub fn layout(&self) -> SKFeatureLayout {
        self.layout
    }

    /// The configured column names carrying the requested kind.
    pub(crate) fn column_names_of_kind(
        &self,
        predicate: impl Fn(SKColumnKind) -> bool,
    ) -> Vec<String> {
        self.columns
            .iter()
            .filter(|(_, kind)| predicate(*kind))
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// A builder that validates the configuration on completion.
#[derive(Debug, Clone, Default)]
pub struct SKTabularSourceConfigBuilder {
    columns: Vec<(String, SKColumnKind)>,
    policy: SKMissingValuePolicy,
    layout: SKFeatureLayout,
}

impl SKTabularSourceConfigBuilder {
    /// Append a column with its kind; order is preserved.
    pub fn column(mut self, name: impl Into<String>, kind: SKColumnKind) -> Self {
        self.columns.push((name.into(), kind));
        self
    }

    /// Apply one missing-value policy to every null in the source.
    pub fn missing_value_policy(mut self, policy: SKMissingValuePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Request a dense or sparse feature block.
    pub fn layout(mut self, layout: SKFeatureLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Finish the configuration.
    ///
    /// # Errors
    /// Returns [`SKError::InvalidHyperparameter`] when no column is selected and
    /// [`SKError::DuplicateColumn`] when a name is repeated.
    pub fn build(self) -> Result<SKTabularSourceConfig, SKError> {
        if self.columns.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "columns",
                reason: "a tabular source config must select at least one column".to_string(),
            });
        }
        for (position, (name, _)) in self.columns.iter().enumerate() {
            if self.columns[..position]
                .iter()
                .any(|(existing, _)| existing == name)
            {
                return Err(SKError::DuplicateColumn {
                    column: name.clone(),
                    suggestion: "declare each column once".to_string(),
                });
            }
        }
        Ok(SKTabularSourceConfig {
            columns: self.columns,
            policy: self.policy,
            layout: self.layout,
        })
    }
}
