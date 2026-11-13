//! The unit of every split: disjoint train and test row indices.

/// A partition of a dataset into train and test rows, expressed as indices.
///
/// The indices address the rows of the original feature matrix, so a split
/// never copies data. `train` and `test` are disjoint, and their union covers
/// the rows a splitter is responsible for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKIndexSplit {
    train: Vec<usize>,
    test: Vec<usize>,
}

impl SKIndexSplit {
    /// Creates an index split from explicitly provided train and test indices.
    pub fn new(train: Vec<usize>, test: Vec<usize>) -> Self {
        Self { train, test }
    }

    /// The train row indices.
    pub fn train(&self) -> &[usize] {
        &self.train
    }

    /// The test row indices.
    pub fn test(&self) -> &[usize] {
        &self.test
    }

    /// The number of train rows.
    pub fn number_of_train_rows(&self) -> usize {
        self.train.len()
    }

    /// The number of test rows.
    pub fn number_of_test_rows(&self) -> usize {
        self.test.len()
    }
}
