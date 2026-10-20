//! Per-column category tables learned by the one-hot encoder.

use std::collections::HashMap;

use sciencekit_common::{
    SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKError, SKLabelTable, sk_canonicalize_labels,
};

/// A reversible integer-code category table: index → code and code → index,
/// assigned in first-occurrence order, excluding the missing sentinel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SKIntegerCategoryTable {
    codes: Vec<i64>,
    index_of: HashMap<i64, usize>,
}

impl SKIntegerCategoryTable {
    /// Register a code, assigning the next index on first occurrence.
    pub(super) fn observe(&mut self, code: i64) {
        if let std::collections::hash_map::Entry::Vacant(entry) = self.index_of.entry(code) {
            entry.insert(self.codes.len());
            self.codes.push(code);
        }
    }

    /// The number of distinct categories.
    pub fn number_of_categories(&self) -> usize {
        self.codes.len()
    }

    /// The index for a code, if present.
    pub fn index_of_code(&self, code: i64) -> Option<usize> {
        self.index_of.get(&code).copied()
    }

    /// The code for an index, if present.
    pub fn code_of(&self, index: usize) -> Option<i64> {
        self.codes.get(index).copied()
    }
}

/// The category table of one input column: nominal (a label table) or integer
/// (a code table). Both are reversible and deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SKOneHotColumnCategories {
    /// A nominal column, backed by an [`SKLabelTable`].
    Nominal(SKLabelTable),
    /// An integer-coded column, backed by an [`SKIntegerCategoryTable`].
    Integer(SKIntegerCategoryTable),
}

impl SKOneHotColumnCategories {
    /// The number of distinct categories in the column.
    pub fn number_of_categories(&self) -> usize {
        match self {
            SKOneHotColumnCategories::Nominal(table) => table.number_of_classes(),
            SKOneHotColumnCategories::Integer(table) => table.number_of_categories(),
        }
    }

    /// The nominal label table, when the column is nominal.
    pub fn as_label_table(&self) -> Option<&SKLabelTable> {
        match self {
            SKOneHotColumnCategories::Nominal(table) => Some(table),
            SKOneHotColumnCategories::Integer(_) => None,
        }
    }

    /// The integer code table, when the column is integer-coded.
    pub fn as_integer_table(&self) -> Option<&SKIntegerCategoryTable> {
        match self {
            SKOneHotColumnCategories::Integer(table) => Some(table),
            SKOneHotColumnCategories::Nominal(_) => None,
        }
    }

    /// The index for an integer code, when the column is integer-coded.
    pub fn index_of_code(&self, code: i64) -> Option<usize> {
        self.as_integer_table()?.index_of_code(code)
    }

    /// The code for an index, when the column is integer-coded.
    pub fn code_of(&self, index: usize) -> Option<i64> {
        self.as_integer_table()?.code_of(index)
    }
}

/// Learn the per-column category tables from a categorical view.
///
/// Nominal columns use `SKLabelTable` interning (first-occurrence order);
/// integer columns skip the missing sentinel. Empty inputs fail before any
/// element is processed.
pub(super) fn build_column_categories(
    view: &SKCategoricalView<'_>,
) -> Result<Vec<SKOneHotColumnCategories>, SKError> {
    if view.is_empty() {
        return Err(SKError::shape_mismatch_2d(1, 1, 0, 0));
    }
    match view {
        SKCategoricalView::Nominal(matrix) => {
            let (rows_count, columns_count) = matrix.shape();
            build_nominal_columns(matrix, rows_count, columns_count)
        }
        SKCategoricalView::Integer(codes) => {
            let (rows_count, columns_count) = codes.dim();
            let mut columns = Vec::with_capacity(columns_count);
            for column in 0..columns_count {
                let mut table = SKIntegerCategoryTable::default();
                for row in 0..rows_count {
                    let code = codes[[row, column]];
                    if code != SK_CATEGORICAL_MISSING_CODE {
                        table.observe(code);
                    }
                }
                columns.push(SKOneHotColumnCategories::Integer(table));
            }
            Ok(columns)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert to a supported categorical representation",
        }),
    }
}

/// Build the nominal columns, validating rectangularity.
fn build_nominal_columns(
    matrix: &sciencekit_common::SKNominalMatrixView<'_>,
    rows_count: usize,
    columns_count: usize,
) -> Result<Vec<SKOneHotColumnCategories>, SKError> {
    for row in 0..rows_count {
        if matrix.row(row).len() != columns_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count, columns_count],
                found: vec![rows_count, matrix.row(row).len()],
            });
        }
    }
    let mut columns = Vec::with_capacity(columns_count);
    for column in 0..columns_count {
        let labels: Vec<&str> = (0..rows_count)
            .map(|row| matrix.value(row, column))
            .collect();
        let (_indices, table) = sk_canonicalize_labels(&labels);
        columns.push(SKOneHotColumnCategories::Nominal(table));
    }
    Ok(columns)
}
