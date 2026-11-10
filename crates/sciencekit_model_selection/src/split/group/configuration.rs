//! The validated group K-fold configuration.

use sciencekit_common::SKExecutionMode;

/// The immutable configuration carried by a built [`super::SKGroupKFold`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKGroupKFoldConfiguration {
    pub(crate) number_of_splits: Option<usize>,
    pub(crate) shuffle: bool,
    pub(crate) random_seed: u64,
    pub(crate) execution_mode: SKExecutionMode,
}
