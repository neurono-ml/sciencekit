//! Contiguous numeric buffers copied across the Lua boundary.
//!
//! Lua tables are not numeric buffers, so every input table is copied into
//! these row-major `f64` buffers and every output is copied back into a table
//! of rows. The structures carry their shape explicitly so the ABI descriptor
//! can be built without re-deriving it.

/// A row-major `f64` matrix.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SKLuaMatrix {
    rows: usize,
    columns: usize,
    values: Vec<f64>,
}

impl SKLuaMatrix {
    /// Wrap values that are already in row-major order.
    pub(crate) fn new(rows: usize, columns: usize, values: Vec<f64>) -> SKLuaMatrix {
        debug_assert_eq!(rows * columns, values.len());
        SKLuaMatrix {
            rows,
            columns,
            values,
        }
    }

    /// The number of rows.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    /// The row-major values.
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }
}

/// A contiguous `f64` vector.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SKLuaVector {
    values: Vec<f64>,
}

impl SKLuaVector {
    /// Wrap a vector of values.
    pub(crate) fn new(values: Vec<f64>) -> SKLuaVector {
        SKLuaVector { values }
    }

    /// The number of values.
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// The values.
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }
}
