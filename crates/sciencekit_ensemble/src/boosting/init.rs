//! Initial raw-score builders and Newton leaf-step computation.

use ndarray::Array2;
use sciencekit_common::SKFloat;

use super::loss::{SKGradientLoss, SKMulticlassLogLoss};

/// Initialize scalar raw scores across all training rows from `loss`.
pub fn sk_initialize_scalar_scores<F: SKFloat, L: SKGradientLoss<F>>(
    loss: &L,
    targets: &[F],
) -> (F, Vec<F>) {
    let initial = loss.initial_prediction(targets);
    (initial, vec![initial; targets.len()])
}

/// Initialize `(rows, C)` raw scores from the class log-priors of `loss`.
pub fn sk_initialize_multiclass_scores<F: SKFloat>(
    loss: &SKMulticlassLogLoss,
    class_indices: &[usize],
) -> (Vec<F>, Array2<F>) {
    let initials = loss.initial_predictions(class_indices);
    let raw_scores = Array2::from_shape_fn(
        (class_indices.len(), loss.number_of_classes()),
        |(_, class_index)| initials[class_index],
    );
    (initials, raw_scores)
}

/// Compute the unshrunk Newton leaf step `-Σg / (Σh + λ)` from first
/// derivatives `gradients` (`g_i = ∂L/∂f`) and `hessians`.
///
/// Degenerate leaves with non-positive Hessian sum keep value zero.
pub fn sk_newton_leaf_value<F: SKFloat>(
    gradients: &[F],
    hessians: &[F],
    l2_regularization: F,
) -> F {
    let negative_gradient_sum = gradients
        .iter()
        .fold(F::zero(), |accumulator, &gradient| accumulator - gradient);
    let hessian_sum = hessians
        .iter()
        .fold(F::zero(), |accumulator, &hessian| accumulator + hessian);
    if hessian_sum <= F::zero() {
        F::zero()
    } else {
        negative_gradient_sum / (hessian_sum + l2_regularization)
    }
}

/// Compute the unshrunk Newton leaf step `Σ(-g) / (Σh + λ)` from negative
/// gradients (`-g_i = -∂L/∂f`) and `hessians`.
pub(crate) fn sk_newton_leaf_value_from_negative_gradients<F: SKFloat>(
    negative_gradients: &[F],
    hessians: &[F],
    l2_regularization: F,
) -> F {
    let negative_gradient_sum = negative_gradients
        .iter()
        .fold(F::zero(), |accumulator, &value| accumulator + value);
    let hessian_sum = hessians
        .iter()
        .fold(F::zero(), |accumulator, &hessian| accumulator + hessian);
    if hessian_sum <= F::zero() {
        F::zero()
    } else {
        negative_gradient_sum / (hessian_sum + l2_regularization)
    }
}
