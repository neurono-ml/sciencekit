//! The binary logistic objective and its L-BFGS solve (spec
//! `logistic-solver`, Decisions 1 and 3).

use ndarray::{Array1, Axis, s};
use sciencekit_common::{SKError, SKFloat};

use super::lbfgs::{SKLbfgsConfig, sk_dot, sk_lbfgs_minimize};
use crate::logistic::design::SKLogisticDesign;
use crate::logistic::objective::{sk_log_sigmoid, sk_sigmoid};

/// A solved binary head: the feature weights, the intercept and the iterations
/// spent.
pub(crate) struct SKBinarySolution<F> {
    /// The feature weights.
    pub(crate) coefficients: Array1<F>,
    /// The intercept.
    pub(crate) intercept: F,
    /// The iterations spent.
    pub(crate) iterations: usize,
}

/// Minimize the (optionally L2-penalized) binary logistic mean loss with the
/// smooth solvers.
///
/// `labels` holds `0.0`/`1.0` and `weights` the per-sample class weights. The
/// point is `[w, b]`, with the intercept last and never penalized.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_solve_binary_head<F, D>(
    design: &D,
    labels: &[F],
    weights: &[F],
    alpha: F,
    l2_penalty: bool,
    include_intercept: bool,
    initial: Option<Array1<F>>,
    config: &SKLbfgsConfig<F>,
) -> Result<SKBinarySolution<F>, SKError>
where
    F: SKFloat,
    D: SKLogisticDesign<F> + Sync,
{
    let columns = design.columns_count();
    let rows = design.rows_count();
    let inverse_rows = F::one() / F::from(rows).expect("row counts fit in the scalar");
    let half = F::from(0.5).expect("0.5 fits");
    let dimension = columns + usize::from(include_intercept);
    let mut point = Array1::<F>::zeros(dimension);
    if let Some(initial) = initial.filter(|value| value.len() == dimension) {
        point.assign(&initial);
    }

    let evaluate = |vector: ndarray::ArrayView1<F>, mut gradient: ndarray::ArrayViewMut1<F>| -> F {
        let (feature_weights, bias) = vector.split_at(Axis(0), columns);
        let mut linear = Array1::<F>::zeros(rows);
        design.dot_all(feature_weights, linear.view_mut());
        if include_intercept {
            let intercept = bias[0];
            linear.mapv_inplace(|value| value + intercept);
        }
        let mut feature_gradient = Array1::<F>::zeros(columns);
        let mut bias_gradient = F::zero();
        let mut loss = F::zero();
        for row in 0..rows {
            let score = linear[row];
            let label = labels[row];
            let sample_weight = weights[row];
            loss = loss
                + sample_weight
                    * (-(label * sk_log_sigmoid(score)
                        + (F::one() - label) * sk_log_sigmoid(-score)));
            let residual = sample_weight * (sk_sigmoid(score) - label) * inverse_rows;
            design.add_row_scaled(row, residual, feature_gradient.view_mut());
            bias_gradient = bias_gradient + residual;
        }
        let mut total = loss * inverse_rows;
        if l2_penalty {
            total = total + half * alpha * sk_dot(feature_weights, feature_weights);
            feature_gradient
                .iter_mut()
                .zip(feature_weights)
                .for_each(|(entry, &weight)| *entry = *entry + alpha * weight);
        }
        gradient.slice_mut(s![..columns]).assign(&feature_gradient);
        if include_intercept {
            gradient[columns] = bias_gradient;
        }
        total
    };

    let (minimizer, iterations) = sk_lbfgs_minimize(evaluate, point, config)?;
    let mut coefficients = Array1::<F>::zeros(columns);
    coefficients.assign(&minimizer.slice(s![..columns]));
    let intercept = if include_intercept {
        minimizer[columns]
    } else {
        F::zero()
    };
    Ok(SKBinarySolution {
        coefficients,
        intercept,
        iterations,
    })
}
