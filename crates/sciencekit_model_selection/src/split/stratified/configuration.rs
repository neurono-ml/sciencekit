//! The validated stratified K-fold configuration.

use sciencekit_common::SKExecutionMode;

/// The immutable configuration carried by a built
/// [`super::SKStratifiedKFold`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKStratifiedKFoldConfiguration {
    pub(crate) number_of_splits: usize,
    pub(crate) shuffle: bool,
    pub(crate) random_seed: u64,
    pub(crate) execution_mode: SKExecutionMode,
}
