//! The SAGA variance-reduced incremental-gradient solver (spec
//! `logistic-solver`, Decision 3).
//!
//! SAGA keeps one stored gradient per sample and applies a proximal step, so it
//! solves the non-smooth L1/ElasticNet objectives and the smooth L2/None
//! objectives through the same iteration. The generic core is objective-agnostic;
//! the wrapper supplies the per-sample and full gradients over a flattened
//! parameter vector.

use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use sciencekit_common::{SKError, SKFloat};

use super::lbfgs::sk_max_abs;
use crate::logistic::config::SKLogisticPenalty;

/// The SAGA convergence controls.
pub(crate) struct SKSagaConfig<F> {
    /// The convergence tolerance on the proximal gradient mapping.
    pub(crate) tolerance: F,
    /// The epoch budget (one epoch = one pass over the samples).
    pub(crate) maximum_number_of_epochs: usize,
    /// The step size.
    pub(crate) step: F,
    /// The deterministic seed of the sample order.
    pub(crate) seed: u64,
}

/// Minimize a composite objective `mean_i f_i(w) + R(w)` with SAGA.
///
/// `sample_gradient(i, point, out)` writes the mean-scaled per-sample gradient
/// `∇f_i/n`; `full_gradient` writes `Σ_i ∇f_i/n`; `penalized` is the number of
/// leading coordinates the proximal operator `R` touches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_saga_minimize<F, S, G>(
    dimension: usize,
    penalized: usize,
    penalty: SKLogisticPenalty,
    alpha: F,
    l1_ratio: F,
    number_of_samples: usize,
    mut sample_gradient: S,
    mut full_gradient: G,
    mut point: Array1<F>,
    config: &SKSagaConfig<F>,
) -> Result<(Array1<F>, usize), SKError>
where
    F: SKFloat,
    S: FnMut(usize, ArrayView1<F>, ArrayViewMut1<F>),
    G: FnMut(ArrayView1<F>, ArrayViewMut1<F>),
{
    let inverse_samples = F::one() / F::from(number_of_samples).expect("counts fit");
    let mut table: Vec<Array1<F>> = Vec::with_capacity(number_of_samples);
    let mut average = Array1::<F>::zeros(dimension);
    for sample in 0..number_of_samples {
        let mut gradient = Array1::<F>::zeros(dimension);
        sample_gradient(sample, point.view(), gradient.view_mut());
        average = average + &gradient;
        table.push(gradient);
    }
    average.mapv_inplace(|value| value * inverse_samples);
    let mut seed = config.seed;
    let mut mapping = Array1::<F>::zeros(dimension);
    for epoch in 0..config.maximum_number_of_epochs {
        let order = sk_permutation(number_of_samples, &mut seed);
        for &sample in &order {
            let mut new_gradient = Array1::<F>::zeros(dimension);
            sample_gradient(sample, point.view(), new_gradient.view_mut());
            for coordinate in 0..dimension {
                let updated = point[coordinate]
                    - config.step
                        * (new_gradient[coordinate] - table[sample][coordinate]
                            + average[coordinate]);
                point[coordinate] = if coordinate < penalized {
                    sk_proximal(penalty, alpha, l1_ratio, config.step, updated)
                } else {
                    updated
                };
            }
            for coordinate in 0..dimension {
                average[coordinate] = average[coordinate]
                    + (new_gradient[coordinate] - table[sample][coordinate]) * inverse_samples;
            }
            table[sample] = new_gradient;
        }
        sk_gradient_mapping(
            &mut full_gradient,
            &point,
            penalized,
            penalty,
            alpha,
            l1_ratio,
            config.step,
            mapping.view_mut(),
        );
        if sk_max_abs(mapping.view()) <= config.tolerance {
            return Ok((point, epoch + 1));
        }
    }
    Err(SKError::not_converged(config.maximum_number_of_epochs))
}

/// The proximal operator of the configured penalty.
pub(crate) fn sk_proximal<F: SKFloat>(
    penalty: SKLogisticPenalty,
    alpha: F,
    l1_ratio: F,
    step: F,
    value: F,
) -> F {
    match penalty {
        SKLogisticPenalty::L1 => sk_soft_threshold(value, step * alpha),
        SKLogisticPenalty::L2 => value / (F::one() + step * alpha),
        SKLogisticPenalty::ElasticNet => {
            sk_soft_threshold(value, step * alpha * l1_ratio)
                / (F::one() + step * alpha * (F::one() - l1_ratio))
        }
        SKLogisticPenalty::None => value,
    }
}

/// The soft-threshold operator `S(z, γ) = sign(z)·max(|z| − γ, 0)`.
fn sk_soft_threshold<F: SKFloat>(value: F, threshold: F) -> F {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        F::zero()
    }
}

/// Write the proximal gradient mapping `(w − prox(w − step·∇F(w))) / step`.
#[allow(clippy::too_many_arguments)]
fn sk_gradient_mapping<F, G>(
    full_gradient: &mut G,
    point: &Array1<F>,
    penalized: usize,
    penalty: SKLogisticPenalty,
    alpha: F,
    l1_ratio: F,
    step: F,
    mut mapping: ArrayViewMut1<F>,
) where
    F: SKFloat,
    G: FnMut(ArrayView1<F>, ArrayViewMut1<F>),
{
    let mut gradient = Array1::<F>::zeros(point.len());
    full_gradient(point.view(), gradient.view_mut());
    for coordinate in 0..point.len() {
        let stepped = point[coordinate] - step * gradient[coordinate];
        let updated = if coordinate < penalized {
            sk_proximal(penalty, alpha, l1_ratio, step, stepped)
        } else {
            stepped
        };
        mapping[coordinate] = (point[coordinate] - updated) / step;
    }
}

/// A deterministic Fisher–Yates permutation of `0..length` (xorshift64*).
fn sk_permutation(length: usize, seed: &mut u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..length).collect();
    for index in (1..length).rev() {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let swap_with = (*seed as usize) % (index + 1);
        order.swap(index, swap_with);
    }
    order
}
