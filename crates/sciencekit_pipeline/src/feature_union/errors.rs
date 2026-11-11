//! Member-indexed error context and probe validation for [`super::SKFeatureUnion`]
//! (spec `feature-union`).

use std::fmt;
use std::io;

use sciencekit_common::SKError;

use crate::SKFeatureBlock;

/// Error returned by [`super::SKFeatureUnion::fit_checked`] and
/// [`super::SKFeatureUnionModel::transform_checked`], pairing the failing
/// zero-based member index (when attributable to a specific member) with the
/// underlying structured [`SKError`].
#[derive(Debug)]
pub struct SKFeatureUnionMemberError {
    member_index: Option<usize>,
    error: SKError,
}

impl SKFeatureUnionMemberError {
    /// Construct an error attributed to `member_index`.
    pub fn member(member_index: usize, error: SKError) -> Self {
        Self {
            member_index: Some(member_index),
            error,
        }
    }

    /// Construct a union-level error not tied to one member.
    pub fn union_level(error: SKError) -> Self {
        Self {
            member_index: None,
            error,
        }
    }

    /// Zero-based index of the member that failed, if the failure came from a
    /// member stage.
    pub fn member_index(&self) -> Option<usize> {
        self.member_index
    }

    /// Borrow the underlying structured [`SKError`].
    pub fn error(&self) -> &SKError {
        &self.error
    }

    /// Convert into [`SKError`], preserving the exact error variant and
    /// prefixing string-carrying messages with the member index.
    pub fn into_sk_error(self) -> SKError {
        match self.member_index {
            Some(index) => sk_wrap_member_error(index, self.error),
            None => self.error,
        }
    }
}

impl fmt::Display for SKFeatureUnionMemberError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.member_index {
            Some(index) => write!(formatter, "feature union member {index}: {}", self.error),
            None => fmt::Display::fmt(&self.error, formatter),
        }
    }
}

impl std::error::Error for SKFeatureUnionMemberError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl From<SKFeatureUnionMemberError> for SKError {
    fn from(value: SKFeatureUnionMemberError) -> Self {
        value.into_sk_error()
    }
}

/// Preserve the structured [`SKError`] variant while attaching the zero-based
/// member index to every variant that carries a dynamic message.
pub(super) fn sk_wrap_member_error(member_index: usize, error: SKError) -> SKError {
    match error {
        SKError::InvalidHyperparameter { name, reason } => SKError::InvalidHyperparameter {
            name,
            reason: format!("feature union member {member_index}: {reason}"),
        },
        SKError::UnsupportedOperation { operation, reason } => SKError::UnsupportedOperation {
            operation,
            reason: format!("feature union member {member_index}: {reason}"),
        },
        SKError::Conversion(message) => {
            SKError::Conversion(format!("feature union member {member_index}: {message}"))
        }
        SKError::Io(io_error) => SKError::Io(io::Error::new(
            io_error.kind(),
            format!("feature union member {member_index}: {io_error}"),
        )),
        SKError::SchemaMismatch { expected, found } => SKError::SchemaMismatch {
            expected,
            found: format!("feature union member {member_index}: {found}"),
        },
        other => other,
    }
}

/// Validate one member's probe output during `fit` (representation and row
/// alignment) and return its column width.
pub(super) fn sk_validate_probe_output<F>(
    member_index: usize,
    probe_output: &SKFeatureBlock<F>,
    expected_rows: usize,
) -> Result<usize, SKFeatureUnionMemberError> {
    match probe_output {
        SKFeatureBlock::Dense(_) => {}
        SKFeatureBlock::Sparse(matrix) if !matrix.is_csr() => {
            return Err(SKFeatureUnionMemberError::member(
                member_index,
                SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert sparse member output to CSR with to_csr",
                },
            ));
        }
        SKFeatureBlock::Sparse(_) => {}
        _ => {
            return Err(SKFeatureUnionMemberError::member(
                member_index,
                SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "member transformers must return a dense or CSR feature block",
                },
            ));
        }
    }
    let found_rows = probe_output.number_of_rows();
    let width = probe_output.number_of_columns();
    if found_rows != expected_rows {
        return Err(SKFeatureUnionMemberError::member(
            member_index,
            SKError::ShapeMismatch {
                expected: vec![expected_rows, width],
                found: vec![found_rows, width],
            },
        ));
    }
    Ok(width)
}
