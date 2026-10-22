//! The multinomial (softmax) logistic objective and its L-BFGS solve (spec
//! `logistic-solver`, Decisions 1 and 3).

use ndarray::{Array1, Array2, s};
use sciencekit_common::{SKError, SKFloat};

use super::lbfgs::{SKLbfgsConfig, sk_dot, sk_lbfgs_minimize};
use crate::logistic::design::SKLogisticDesign;
use crate::logistic::objective::{sk_multinomial_loss, sk_softmax};

/// A solved multinomial head set: the per-class weights, the per-class
/// intercepts and the iterations spent.
pub(crate) struct SKMultinomialSolution<F> {
    /// The per-class weights, shaped `classes × features`.
    pub(crate) coefficients: Array2<F>,
    /// The per-class intercepts.
    pub(crate) intercepts: Array1<F>,
    /// The iterations spent.
    pub(crate) iterations: usize,
}

/// Minimize the (optionally L2-penalized) multinomial softmax loss with the
/// smooth solvers.
///
/// The point flattens `[W (row-major), b]`; the intercepts are never penalized
/// and the L2 penalty is applied per class.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_solve_multinomial<F, D>(
    design: &D,
    labels: &[usize],
    weights: &[F],
    number_of_classes: usize,
    alpha: F,
    l2_penalty: bool,
    include_intercept: bool,
    initial: Option<Array1<F>>,
    config: &SKLbfgsConfig<F>,
) -> Result<SKMultinomialSolution<F>, SKError>
where
    F: SKFloat,
    D: SKLogisticDesign<F> + Sync,
{
    let columns = design.columns_count();
    let rows = design.rows_count();
    let heads = number_of_classes;
    let inverse_rows = F::one() / F::from(rows).expect("row counts fit in the scalar");
    let half = F::from(0.5).expect("0.5 fits");
    let weight_dimension = heads * columns;
    let dimension = weight_dimension + usize::from(include_intercept) * heads;
    let mut point = Array1::<F>::zeros(dimension);
    if let Some(initial) = initial.filter(|value| value.len() == dimension) {
        point.assign(&initial);
    }

    let evaluate = |vector: ndarray::ArrayView1<F>, mut gradient: ndarray::ArrayViewMut1<F>| -> F {
        let mut linear = Array2::<F>::zeros((rows, heads));
        for head in 0..heads {
            let head_weights = vector.slice(s![head * columns..(head + 1) * columns]);
            design.dot_all(head_weights, linear.column_mut(head));
        }
        if include_intercept {
            let intercepts = vector.slice(s![weight_dimension..]);
            for mut row in linear.rows_mut() {
                row.zip_mut_with(&intercepts, |entry, &value| *entry = *entry + value);
            }
        }
        let mut feature_gradient = Array2::<F>::zeros((heads, columns));
        let mut intercept_gradient = Array1::<F>::zeros(heads);
        let mut loss = F::zero();
        let mut logits = vec![F::zero(); heads];
        let mut probabilities = vec![F::zero(); heads];
        let mut class_gradient = vec![F::zero(); heads];
        for row in 0..rows {
            for (head, entry) in logits.iter_mut().enumerate() {
                *entry = linear[[row, head]];
            }
            let label = labels[row];
            loss = loss + weights[row] * sk_multinomial_loss(&logits, label, F::one());
            sk_softmax(&logits, &mut probabilities);
            class_gradient
                .iter_mut()
                .enumerate()
                .for_each(|(head, entry)| {
                    let indicator = if head == label { F::one() } else { F::zero() };
                    *entry = weights[row] * (probabilities[head] - indicator) * inverse_rows;
                });
            for head in 0..heads {
                design.add_row_scaled(row, class_gradient[head], feature_gradient.row_mut(head));
                intercept_gradient[head] = intercept_gradient[head] + class_gradient[head];
            }
        }
        let mut total = loss * inverse_rows;
        if l2_penalty {
            for head in 0..heads {
                let head_weights = vector.slice(s![head * columns..(head + 1) * columns]);
                total = total + half * alpha * sk_dot(head_weights, head_weights);
                feature_gradient
                    .row_mut(head)
                    .iter_mut()
                    .zip(head_weights)
                    .for_each(|(entry, &weight)| *entry = *entry + alpha * weight);
            }
        }
        for head in 0..heads {
            gradient
                .slice_mut(s![head * columns..(head + 1) * columns])
                .assign(&feature_gradient.row(head));
        }
        if include_intercept {
            gradient
                .slice_mut(s![weight_dimension..])
                .assign(&intercept_gradient);
        }
        total
    };

    let (minimizer, iterations) = sk_lbfgs_minimize(evaluate, point, config)?;
    let mut coefficients = Array2::<F>::zeros((heads, columns));
    for head in 0..heads {
        coefficients
            .row_mut(head)
            .assign(&minimizer.slice(s![head * columns..(head + 1) * columns]));
    }
    let intercepts = if include_intercept {
        minimizer.slice(s![weight_dimension..]).to_owned()
    } else {
        Array1::zeros(heads)
    };
    Ok(SKMultinomialSolution {
        coefficients,
        intercepts,
        iterations,
    })
}
