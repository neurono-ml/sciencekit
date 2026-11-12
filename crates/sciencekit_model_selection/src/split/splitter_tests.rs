//! The splitter contract accepts optional targets and groups.

use ndarray::{Array1, Array2, ArrayView1};
use sciencekit_common::{SKDataView, SKError, SKFloat, SKTargetView};

use super::SKSplitter;
use crate::SKIndexSplit;

/// A minimal splitter used to exercise the contract surface.
struct SKStubSplitter;

impl<F: SKFloat> SKSplitter<F> for SKStubSplitter {
    fn split(
        &self,
        features: SKDataView<'_, F>,
        _targets: Option<SKTargetView<'_, F>>,
        groups: Option<ArrayView1<'_, i64>>,
    ) -> Result<Vec<SKIndexSplit>, SKError> {
        let number_of_rows = match features {
            SKDataView::Dense(view) => view.nrows(),
            SKDataView::Sparse(view) => view.rows(),
            _ => unreachable!("the contract currently has two representations"),
        };
        let observed_groups = groups.map(|values| values.len()).unwrap_or(0);
        Ok(vec![SKIndexSplit::new(
            (0..number_of_rows).collect(),
            (0..observed_groups).collect(),
        )])
    }
}

#[test]
fn splitter_contract_accepts_optional_targets_and_groups() {
    let features = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let targets = Array1::from_vec(vec![0_i64, 1, 0]);
    let groups = Array1::from_vec(vec![7_i64, 8, 9]);

    let splits = SKStubSplitter
        .split(
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Integer(targets.view())),
            Some(groups.view()),
        )
        .expect("the stub splitter succeeds");

    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].train(), &[0, 1, 2]);
    assert_eq!(splits[0].test(), &[0, 1, 2]);
}

#[test]
fn splitter_contract_accepts_a_split_without_targets_or_groups() {
    let features = Array2::<f64>::zeros((2, 1));

    let splits = SKStubSplitter
        .split(SKDataView::Dense(features.view()), None, None)
        .expect("the stub splitter succeeds");

    assert_eq!(splits[0].train().len(), 2);
    assert!(splits[0].test().is_empty());
}
