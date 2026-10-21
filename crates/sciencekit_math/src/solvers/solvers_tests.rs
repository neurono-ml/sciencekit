//! TDD tests for the iterative linear-solver substrate (spec
//! `iterative-linear-solvers`).

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, array};
use rayon::prelude::*;
use sciencekit_common::{SKError, SKFloat};

use super::{SKLinearOperator, sk_conjugate_gradient};

/// Assert two `f64` vectors agree within a relative tolerance.
pub(super) fn assert_close_1d(actual: &Array1<f64>, expected: &Array1<f64>, tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "values {a} and {e} differ beyond tolerance {tolerance}"
        );
    }
}

/// A test-only dense linear operator; `parallel` selects a rayon application.
struct SKTestDenseOperator<F: SKFloat> {
    matrix: Array2<F>,
    parallel: bool,
}

impl<F: SKFloat> SKTestDenseOperator<F> {
    /// Build a sequential test operator over a dense matrix.
    fn new(matrix: Array2<F>) -> Self {
        Self {
            matrix,
            parallel: false,
        }
    }

    /// Build a test operator whose application runs across the compute pool.
    fn parallel(matrix: Array2<F>) -> Self {
        Self {
            matrix,
            parallel: true,
        }
    }
}

impl<F: SKFloat> SKLinearOperator<F> for SKTestDenseOperator<F> {
    fn dimension(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply_into(&self, input: ArrayView1<F>, mut output: ArrayViewMut1<F>) {
        if self.parallel {
            let owned = input.to_owned();
            let rows_count = self.matrix.nrows();
            let values: Vec<F> = (0..rows_count)
                .into_par_iter()
                .map(|row_index| self.matrix.row(row_index).dot(&owned))
                .collect();
            output.assign(&Array1::from_vec(values));
        } else {
            output.assign(&self.matrix.dot(&input));
        }
    }
}

/// The operator reports its dimension and writes its product without mutating
/// the input or its own borrowed data.
#[test]
fn dense_operator_reports_dimension_and_writes_product() {
    let matrix: Array2<f64> = array![[4.0, 1.0], [1.0, 3.0]];
    let operator = SKTestDenseOperator::new(matrix);
    assert_eq!(operator.dimension(), 2);
    let input = array![1.0, 2.0];
    let mut output = Array1::zeros(2);
    operator.apply_into(input.view(), output.view_mut());
    assert_eq!(output, array![6.0, 7.0]);
    assert_eq!(input, array![1.0, 2.0]);
    assert_eq!(operator.apply(input.view()), array![6.0, 7.0]);
}

/// Conjugate gradient solves the hand-checkable SPD system
/// `[[4, 1], [1, 3]] x = [1, 2]`, whose exact solution is `[1/11, 7/11]`.
#[test]
fn conjugate_gradient_solves_hand_checkable_system() {
    let operator = SKTestDenseOperator::new(array![[4.0, 1.0], [1.0, 3.0]]);
    let solution = sk_conjugate_gradient(&operator, array![1.0, 2.0].view(), 1e-12, 10)
        .expect("SPD system converges");
    assert_close_1d(&solution, &array![1.0 / 11.0, 7.0 / 11.0], 1e-9);
}

/// Conjugate gradient agrees with a direct solve on a well-conditioned SPD
/// system.
#[test]
fn conjugate_gradient_matches_a_direct_solve() {
    let matrix: Array2<f64> = array![[2.0, -1.0], [-1.0, 2.0]];
    let right_hand_side = array![1.0, 0.0];
    // Direct solve via Cramer's rule (the well-conditioned 2×2 reference).
    let determinant = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
    let direct = array![
        (right_hand_side[0] * matrix[[1, 1]] - matrix[[0, 1]] * right_hand_side[1]) / determinant,
        (matrix[[0, 0]] * right_hand_side[1] - right_hand_side[0] * matrix[[1, 0]]) / determinant,
    ];
    let operator = SKTestDenseOperator::new(matrix);
    let solution =
        sk_conjugate_gradient(&operator, right_hand_side.view(), 1e-12, 10).expect("converges");
    assert_close_1d(&solution, &direct, 1e-9);
}

/// The returned solution drives the residual below the requested tolerance.
#[test]
fn conjugate_gradient_drives_the_residual_below_tolerance() {
    let matrix: Array2<f64> = array![[4.0, 1.0], [1.0, 3.0]];
    let right_hand_side = array![1.0, 2.0];
    let operator = SKTestDenseOperator::new(matrix.clone());
    let solution =
        sk_conjugate_gradient(&operator, right_hand_side.view(), 1e-10, 10).expect("converges");
    let residual = matrix.dot(&solution) - &right_hand_side;
    assert!(residual.dot(&residual).sqrt() <= 1e-10);
}

/// A race-free parallel operator application matches the sequential one.
#[test]
fn parallel_operator_apply_matches_sequential() {
    let matrix: Array2<f64> = Array2::from_shape_fn((40, 40), |(i, j)| {
        if i == j {
            4.0
        } else {
            0.1 / (1.0 + (i + j) as f64)
        }
    });
    let input = Array1::from_shape_fn(40, |i| (i as f64) * 0.5 - 1.0);
    let sequential = SKTestDenseOperator::new(matrix.clone()).apply(input.view());
    let parallel = SKTestDenseOperator::parallel(matrix).apply(input.view());
    assert_close_1d(&parallel, &sequential, 1e-9);
}

/// A zero iteration budget fails immediately, reporting zero iterations.
#[test]
fn zero_iteration_budget_fails_immediately() {
    let operator = SKTestDenseOperator::new(array![[4.0, 1.0], [1.0, 3.0]]);
    let result = sk_conjugate_gradient(&operator, array![1.0, 2.0].view(), 1e-12, 0);
    match result {
        Err(SKError::NotConverged { iterations }) => assert_eq!(iterations, 0),
        other => panic!("expected NotConverged, got {other:?}"),
    }
}

/// An unreachable tolerance within a small budget reports the budget spent.
#[test]
fn unreachable_tolerance_reports_the_budget() {
    let operator = SKTestDenseOperator::new(array![[4.0, 1.0], [1.0, 3.0]]);
    let result = sk_conjugate_gradient(&operator, array![1.0, 2.0].view(), 1e-30, 1);
    match result {
        Err(SKError::NotConverged { iterations }) => assert_eq!(iterations, 1),
        other => panic!("expected NotConverged, got {other:?}"),
    }
}

/// A zero right-hand side solves to the zero vector with a non-zero budget.
#[test]
fn zero_right_hand_side_solves_to_zero() {
    let operator = SKTestDenseOperator::new(array![[4.0, 1.0], [1.0, 3.0]]);
    let solution =
        sk_conjugate_gradient(&operator, array![0.0, 0.0].view(), 1e-12, 10).expect("converges");
    assert_eq!(solution, array![0.0, 0.0]);
}

/// The solve is generic over the two supported floats.
#[test]
fn conjugate_gradient_accepts_either_float() {
    fn solve<F: SKFloat>(matrix: Array2<F>, right_hand_side: Array1<F>, tolerance: F) -> Array1<F> {
        let operator = SKTestDenseOperator::new(matrix);
        sk_conjugate_gradient(&operator, right_hand_side.view(), tolerance, 10).expect("converges")
    }
    let x64 = solve(
        array![[4.0_f64, 1.0], [1.0, 3.0]],
        array![1.0_f64, 2.0],
        1e-12,
    );
    assert_close_1d(&x64, &array![1.0 / 11.0, 7.0 / 11.0], 1e-9);
    let x32 = solve(
        array![[4.0_f32, 1.0], [1.0, 3.0]],
        array![1.0_f32, 2.0],
        1e-5,
    );
    assert!((x32[0] - 1.0 / 11.0).abs() < 1e-4);
    assert!((x32[1] - 7.0 / 11.0).abs() < 1e-4);
}
