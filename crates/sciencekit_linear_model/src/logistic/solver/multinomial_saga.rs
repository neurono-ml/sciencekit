//! The multinomial logistic objective solved with SAGA (spec
//! `logistic-solver`, Decision 3).

use ndarray::{Array1, s};
use sciencekit_common::{SKError, SKFloat};

use super::saga::{SKSagaConfig, sk_saga_minimize};
use crate::logistic::config::SKLogisticConfig;
use crate::logistic::design::SKLogisticDesign;
use crate::logistic::objective::sk_softmax;

const SAGA_SEED: u64 = 0x1319_8A2E_0370_7344;

/// Minimize the multinomial logistic objective with SAGA, returning the
/// per-class weights, the per-class intercepts and the epochs spent.
pub(crate) fn sk_solve_multinomial_saga<F, D>(
    design: &D,
    labels: &[usize],
    weights: &[F],
    number_of_classes: usize,
    initial: Option<Array1<F>>,
    config: &SKLogisticConfig<F>,
) -> Result<(ndarray::Array2<F>, Array1<F>, usize), SKError>
where
    F: SKFloat,
    D: SKLogisticDesign<F> + Sync,
{
    let columns = design.columns_count();
    let rows = design.rows_count();
    let heads = number_of_classes;
    let inverse_rows = F::one() / F::from(rows).expect("row counts fit");
    let weight_dimension = heads * columns;
    let dimension = weight_dimension + usize::from(config.include_intercept) * heads;
    let half = F::from(0.5).expect("fits");
    let lipschitz = (0..rows)
        .map(|row| weights[row] * design.row_squared_norm(row))
        .fold(F::zero(), |accumulator, value| accumulator.max(value))
        * half;
    let step =
        F::from(0.5).expect("fits") / (lipschitz + config.alpha + F::from(1e-12).expect("fits"));
    let saga_config = SKSagaConfig {
        tolerance: config.tolerance,
        maximum_number_of_epochs: config.maximum_number_of_iterations,
        step,
        seed: SAGA_SEED,
    };

    let accumulate =
        |sample: usize, point: ndarray::ArrayView1<F>, mut out: ndarray::ArrayViewMut1<F>| {
            let intercepts = point.slice(s![weight_dimension..]);
            let mut logits = vec![F::zero(); heads];
            for (head, logit) in logits.iter_mut().enumerate() {
                let head_weights = point.slice(s![head * columns..(head + 1) * columns]);
                *logit = design.row_dot(sample, head_weights);
                if config.include_intercept {
                    *logit = *logit + intercepts[head];
                }
            }
            let mut probabilities = vec![F::zero(); heads];
            sk_softmax(&logits, &mut probabilities);
            out.fill(F::zero());
            for head in 0..heads {
                let indicator = if head == labels[sample] {
                    F::one()
                } else {
                    F::zero()
                };
                let coefficient = weights[sample] * (probabilities[head] - indicator);
                design.add_row_scaled(
                    sample,
                    coefficient,
                    out.slice_mut(s![head * columns..(head + 1) * columns]),
                );
                if config.include_intercept {
                    out[weight_dimension + head] = coefficient;
                }
            }
        };
    let full_gradient = |point: ndarray::ArrayView1<F>, mut out: ndarray::ArrayViewMut1<F>| {
        out.fill(F::zero());
        for row in 0..rows {
            let intercepts = point.slice(s![weight_dimension..]);
            let mut logits = vec![F::zero(); heads];
            for (head, logit) in logits.iter_mut().enumerate() {
                let head_weights = point.slice(s![head * columns..(head + 1) * columns]);
                *logit = design.row_dot(row, head_weights);
                if config.include_intercept {
                    *logit = *logit + intercepts[head];
                }
            }
            let mut probabilities = vec![F::zero(); heads];
            sk_softmax(&logits, &mut probabilities);
            for head in 0..heads {
                let indicator = if head == labels[row] {
                    F::one()
                } else {
                    F::zero()
                };
                let coefficient = weights[row] * (probabilities[head] - indicator) * inverse_rows;
                design.add_row_scaled(
                    row,
                    coefficient,
                    out.slice_mut(s![head * columns..(head + 1) * columns]),
                );
                if config.include_intercept {
                    out[weight_dimension + head] = out[weight_dimension + head] + coefficient;
                }
            }
        }
    };

    let (minimizer, epochs) = sk_saga_minimize(
        dimension,
        weight_dimension,
        config.penalty,
        config.alpha,
        config.l1_ratio,
        rows,
        accumulate,
        full_gradient,
        initial.unwrap_or_else(|| Array1::<F>::zeros(dimension)),
        &saga_config,
    )?;
    let mut coefficients = ndarray::Array2::<F>::zeros((heads, columns));
    for head in 0..heads {
        coefficients
            .row_mut(head)
            .assign(&minimizer.slice(s![head * columns..(head + 1) * columns]));
    }
    let intercepts = if config.include_intercept {
        minimizer.slice(s![weight_dimension..]).to_owned()
    } else {
        Array1::zeros(heads)
    };
    Ok((coefficients, intercepts, epochs))
}
