//! The binary logistic objective solved with SAGA (spec `logistic-solver`,
//! Decision 3).

use ndarray::{Array1, Axis, s};
use sciencekit_common::{SKError, SKFloat};

use super::saga::{SKSagaConfig, sk_saga_minimize};
use crate::logistic::config::SKLogisticConfig;
use crate::logistic::design::SKLogisticDesign;
use crate::logistic::objective::sk_sigmoid;

/// The deterministic seed shared by the SAGA fits.
const SAGA_SEED: u64 = 0x243F_6A88_85A3_08D3;

/// Minimize the binary logistic objective with SAGA, returning the weights, the
/// intercept and the epochs spent.
pub(crate) fn sk_solve_binary_head_saga<F, D>(
    design: &D,
    labels: &[F],
    weights: &[F],
    initial: Option<Array1<F>>,
    config: &SKLogisticConfig<F>,
) -> Result<(Array1<F>, F, usize), SKError>
where
    F: SKFloat,
    D: SKLogisticDesign<F> + Sync,
{
    let columns = design.columns_count();
    let rows = design.rows_count();
    let inverse_rows = F::one() / F::from(rows).expect("row counts fit");
    let dimension = columns + usize::from(config.include_intercept);
    let quarter = F::from(0.25).expect("fits");
    let lipschitz = (0..rows)
        .map(|row| weights[row] * design.row_squared_norm(row))
        .fold(F::zero(), |accumulator, value| accumulator.max(value))
        * quarter;
    let step =
        F::from(0.5).expect("fits") / (lipschitz + config.alpha + F::from(1e-12).expect("fits"));
    let saga_config = SKSagaConfig {
        tolerance: config.tolerance,
        maximum_number_of_epochs: config.maximum_number_of_iterations,
        step,
        seed: SAGA_SEED,
    };

    let sample_gradient =
        |sample: usize, point: ndarray::ArrayView1<F>, mut out: ndarray::ArrayViewMut1<F>| {
            let (feature_weights, bias) = point.split_at(Axis(0), columns);
            let mut score = design.row_dot(sample, feature_weights);
            if config.include_intercept {
                score = score + bias[0];
            }
            let coefficient = weights[sample] * (sk_sigmoid(score) - labels[sample]);
            out.fill(F::zero());
            design.add_row_scaled(sample, coefficient, out.slice_mut(s![..columns]));
            if config.include_intercept {
                out[columns] = coefficient;
            }
        };
    let full_gradient = |point: ndarray::ArrayView1<F>, mut out: ndarray::ArrayViewMut1<F>| {
        let (feature_weights, bias) = point.split_at(Axis(0), columns);
        out.fill(F::zero());
        for row in 0..rows {
            let mut score = design.row_dot(row, feature_weights);
            if config.include_intercept {
                score = score + bias[0];
            }
            let coefficient = weights[row] * (sk_sigmoid(score) - labels[row]) * inverse_rows;
            design.add_row_scaled(row, coefficient, out.slice_mut(s![..columns]));
            if config.include_intercept {
                out[columns] = out[columns] + coefficient;
            }
        }
    };

    let (minimizer, epochs) = sk_saga_minimize(
        dimension,
        columns,
        config.penalty,
        config.alpha,
        config.l1_ratio,
        rows,
        sample_gradient,
        full_gradient,
        initial.unwrap_or_else(|| Array1::<F>::zeros(dimension)),
        &saga_config,
    )?;
    let mut coefficients = Array1::<F>::zeros(columns);
    coefficients.assign(&minimizer.slice(s![..columns]));
    let intercept = if config.include_intercept {
        minimizer[columns]
    } else {
        F::zero()
    };
    Ok((coefficients, intercept, epochs))
}
