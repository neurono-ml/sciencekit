//! Shape and value readers shared by the mock estimators.

use sciencekit_common::SKDataView;

pub(super) fn sk_number_of_rows(features: SKDataView<'_, f64>) -> usize {
    match features {
        SKDataView::Dense(view) => view.nrows(),
        SKDataView::Sparse(view) => view.rows(),
        _ => 0,
    }
}

pub(super) fn sk_first_column(features: SKDataView<'_, f64>, row: usize) -> f64 {
    match features {
        SKDataView::Dense(view) => view[[row, 0]],
        SKDataView::Sparse(view) => view.get(row, 0).copied().unwrap_or(0.0),
        _ => 0.0,
    }
}
