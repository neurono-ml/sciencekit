//! Node-attributed error context and feature-block validation for
//! [`super::SKPipelineGraph`] (spec `pipeline-graph`).

use std::fmt;
use std::io;

use sciencekit_common::SKError;

use crate::SKFeatureBlock;

use super::node::SKNodeIdentifier;

/// Error returned by checked DAG execution methods, pairing the failing
/// [`SKNodeIdentifier`] (when attributable to a specific graph node) with the
/// underlying structured [`SKError`].
#[derive(Debug)]
pub struct SKPipelineGraphNodeError {
    node_identifier: Option<SKNodeIdentifier>,
    error: SKError,
}

impl SKPipelineGraphNodeError {
    /// Construct an error attributed to `node_identifier`.
    pub fn node(node_identifier: SKNodeIdentifier, error: SKError) -> Self {
        Self {
            node_identifier: Some(node_identifier),
            error,
        }
    }

    /// Construct a graph-level error not tied to one specific node.
    pub fn graph_level(error: SKError) -> Self {
        Self {
            node_identifier: None,
            error,
        }
    }

    /// Identifier of the failing node, when the failure came from one node.
    pub fn node_identifier(&self) -> Option<SKNodeIdentifier> {
        self.node_identifier
    }

    /// Borrow the underlying structured [`SKError`].
    pub fn error(&self) -> &SKError {
        &self.error
    }

    /// Convert into [`SKError`], preserving structured error variants and
    /// prefixing dynamic-message variants with the node identifier.
    pub fn into_sk_error(self) -> SKError {
        match self.node_identifier {
            Some(identifier) => sk_wrap_node_error(identifier, self.error),
            None => self.error,
        }
    }
}

impl fmt::Display for SKPipelineGraphNodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node_identifier {
            Some(identifier) => write!(formatter, "pipeline graph {identifier}: {}", self.error),
            None => fmt::Display::fmt(&self.error, formatter),
        }
    }
}

impl std::error::Error for SKPipelineGraphNodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl From<SKError> for SKPipelineGraphNodeError {
    fn from(error: SKError) -> Self {
        Self::graph_level(error)
    }
}

impl From<SKPipelineGraphNodeError> for SKError {
    fn from(value: SKPipelineGraphNodeError) -> Self {
        value.into_sk_error()
    }
}

fn sk_wrap_node_error(node_identifier: SKNodeIdentifier, error: SKError) -> SKError {
    let prefix = format!("pipeline graph {node_identifier}: ");
    match error {
        SKError::InvalidHyperparameter { name, reason } => SKError::InvalidHyperparameter {
            name,
            reason: format!("{prefix}{reason}"),
        },
        SKError::UnsupportedOperation { operation, reason } => SKError::UnsupportedOperation {
            operation,
            reason: format!("{prefix}{reason}"),
        },
        SKError::Conversion(message) => SKError::Conversion(format!("{prefix}{message}")),
        SKError::Io(io_error) => SKError::Io(io::Error::new(
            io_error.kind(),
            format!("{prefix}{io_error}"),
        )),
        SKError::UnknownColumn { column, suggestion } => SKError::UnknownColumn {
            column,
            suggestion: format!("{prefix}{suggestion}"),
        },
        SKError::DuplicateColumn { column, suggestion } => SKError::DuplicateColumn {
            column,
            suggestion: format!("{prefix}{suggestion}"),
        },
        SKError::UnsupportedColumnType {
            column,
            logical_type,
            suggestion,
        } => SKError::UnsupportedColumnType {
            column,
            logical_type,
            suggestion: format!("{prefix}{suggestion}"),
        },
        SKError::SchemaMismatch { expected, found } => SKError::SchemaMismatch {
            expected,
            found: format!("{prefix}{found}"),
        },
        other => other,
    }
}

/// Validate a block's representation (dense or CSR) and row count for `node`.
pub(super) fn sk_validate_node_block<F>(
    node: Option<SKNodeIdentifier>,
    block: &SKFeatureBlock<F>,
    expected_rows: usize,
) -> Result<usize, SKPipelineGraphNodeError> {
    let wrap = |error: SKError| match node {
        Some(identifier) => SKPipelineGraphNodeError::node(identifier, error),
        None => SKPipelineGraphNodeError::graph_level(error),
    };
    match block {
        SKFeatureBlock::Dense(_) => {}
        SKFeatureBlock::Sparse(matrix) if !matrix.is_csr() => {
            return Err(wrap(SKError::UnsupportedRepresentation {
                representation: "csc",
                suggestion: "convert sparse matrix or node output to CSR with to_csr",
            }));
        }
        SKFeatureBlock::Sparse(_) => {}
        _ => {
            return Err(wrap(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "graph nodes must exchange dense or CSR feature blocks",
            }));
        }
    }
    let found_rows = block.number_of_rows();
    let width = block.number_of_columns();
    if found_rows != expected_rows {
        return Err(wrap(SKError::ShapeMismatch {
            expected: vec![expected_rows, width],
            found: vec![found_rows, width],
        }));
    }
    Ok(width)
}
