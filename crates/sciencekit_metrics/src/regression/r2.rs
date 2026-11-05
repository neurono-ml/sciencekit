//! The coefficient of determination.

use ndarray::ArrayView1;
use sciencekit_common::{SKError, SKFloat};

use super::inputs::sk_validate_regression_inputs;
use super::sums::SKCompensatedSum;

/// The coefficient of determination `R² = 1 − SSE / SST`.
///
/// Constant targets follow the documented finite policy: one when the residual
/// is zero and zero otherwise, never NaN.
pub fn sk_r2_score<F: SKFloat>(
    targets: ArrayView1<'_, F>,
    predictions: ArrayView1<'_, F>,
) -> Result<F, SKError> {
    sk_validate_regression_inputs(&targets, &predictions, "r2")?;
    let mut squared_errors = SKCompensatedSum::new();
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        let error = *target - *prediction;
        squared_errors.add(error * error);
    }
    let residual = squared_errors.value();

    // Constant targets have zero total variance: the finite policy applies.
    let first = targets[0];
    if targets.iter().all(|target| *target == first) {
        return Ok(if residual.is_zero() {
            F::one()
        } else {
            F::zero()
        });
    }

    let mut total_sum = SKCompensatedSum::new();
    for target in targets.iter() {
        total_sum.add(*target);
    }
    let count = num_traits::cast::<usize, F>(targets.len()).expect("counts fit in the scalar");
    let mean = total_sum.value() / count;
    let mut total_variance = SKCompensatedSum::new();
    for target in targets.iter() {
        let deviation = *target - mean;
        total_variance.add(deviation * deviation);
    }
    let total = total_variance.value();
    if total.is_zero() {
        return Ok(if residual.is_zero() {
            F::one()
        } else {
            F::zero()
        });
    }
    Ok(F::one() - residual / total)
}
