//! View dimensions used by the grid-search observability spans.

use sciencekit_common::{SKDataView, SKFloat};

/// The `(rows, columns)` shape of a feature view, whichever representation.
pub(crate) fn sk_view_dimensions<F: SKFloat>(features: SKDataView<'_, F>) -> (usize, usize) {
    match features {
        SKDataView::Dense(view) => (view.nrows(), view.ncols()),
        SKDataView::Sparse(view) => (view.rows(), view.cols()),
        _ => (0, 0),
    }
}
