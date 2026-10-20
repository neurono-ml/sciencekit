//! Per-row category resolution for the fitted one-hot encoder.

use sciencekit_common::{SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKError};

use crate::one_hot_encoder::categories::SKOneHotColumnCategories;
use crate::one_hot_encoder::model::SKOneHotEncoderModel;

impl SKOneHotEncoderModel {
    /// The global output indices defined for `row`: known categories not equal
    /// to the missing sentinel. Unknown categories fail under `Reject` and are
    /// skipped under `Ignore`.
    pub(super) fn resolve_row(
        &self,
        view: &SKCategoricalView<'_>,
        row: usize,
    ) -> Result<Vec<usize>, SKError> {
        let mut output = Vec::new();
        match view {
            SKCategoricalView::Nominal(matrix) => {
                for (column, categories) in self.columns.iter().enumerate() {
                    let label = matrix.value(row, column);
                    match categories {
                        SKOneHotColumnCategories::Nominal(table) => {
                            if let Some(local) = table.index_of(label) {
                                output.push(self.offsets[column] + local);
                            } else if self.handle_unknown.rejects() {
                                return Err(unknown_category("label", label, column));
                            }
                        }
                        SKOneHotColumnCategories::Integer(_) => {
                            return Err(representation_changed("nominal"));
                        }
                    }
                }
            }
            SKCategoricalView::Integer(codes) => {
                for (column, categories) in self.columns.iter().enumerate() {
                    let code = codes[[row, column]];
                    if code == SK_CATEGORICAL_MISSING_CODE {
                        continue;
                    }
                    match categories {
                        SKOneHotColumnCategories::Integer(table) => {
                            if let Some(local) = table.index_of_code(code) {
                                output.push(self.offsets[column] + local);
                            } else if self.handle_unknown.rejects() {
                                return Err(SKError::Conversion(format!(
                                    "unknown code `{code}` in column {column}"
                                )));
                            }
                        }
                        SKOneHotColumnCategories::Nominal(_) => {
                            return Err(representation_changed("integer"));
                        }
                    }
                }
            }
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "transform with a supported categorical representation",
                });
            }
        }
        Ok(output)
    }
}

/// A structured error for an unknown nominal category under `Reject`.
fn unknown_category(kind: &str, value: &str, column: usize) -> SKError {
    SKError::Conversion(format!("unknown {kind} `{value}` in column {column}"))
}

/// The representation used to transform differs from the one used to fit.
fn representation_changed(found: &'static str) -> SKError {
    SKError::UnsupportedRepresentation {
        representation: found,
        suggestion: "fit and transform with the same categorical representation",
    }
}
