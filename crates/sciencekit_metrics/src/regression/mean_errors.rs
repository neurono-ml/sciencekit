//! Mean squared error, root mean squared error and mean absolute error.

use ndarray::ArrayView1;
use sciencekit_common::{SKError, SKFloat};

use super::inputs::sk_validate_regression_inputs;
use super::sums::SKCompensatedSum;

/// The mean of the squared prediction errors.
pub fn sk_mean_squared_error<F: SKFloat>(
    targets: ArrayView1<'_, F>,
    predictions: ArrayView1<'_, F>,
) -> Result<F, SKError> {
    sk_validate_regression_inputs(&targets, &predictions, "mean_squared_error")?;
    Ok(sk_mean_squared_error_unchecked(&targets, &predictions))
}

/// The square root of the mean squared error.
pub fn sk_root_mean_squared_error<F: SKFloat>(
    targets: ArrayView1<'_, F>,
    predictions: ArrayView1<'_, F>,
) -> Result<F, SKError> {
    sk_validate_regression_inputs(&targets, &predictions, "root_mean_squared_error")?;
    Ok(sk_mean_squared_error_unchecked(&targets, &predictions).sqrt())
}

/// The mean of the absolute prediction errors.
pub fn sk_mean_absolute_error<F: SKFloat>(
    targets: ArrayView1<'_, F>,
    predictions: ArrayView1<'_, F>,
) -> Result<F, SKError> {
    sk_validate_regression_inputs(&targets, &predictions, "mean_absolute_error")?;
    Ok(sk_mean_absolute_error_unchecked(&targets, &predictions))
}

/// The mean squared error over already-validated inputs.
pub(super) fn sk_mean_squared_error_unchecked<F: SKFloat>(
    targets: &ArrayView1<F>,
    predictions: &ArrayView1<F>,
) -> F {
    let mut sum = SKCompensatedSum::new();
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        let error = *target - *prediction;
        sum.add(error * error);
    }
    sk_mean(sum.value(), targets.len())
}

/// The mean absolute error over already-validated inputs.
pub(super) fn sk_mean_absolute_error_unchecked<F: SKFloat>(
    targets: &ArrayView1<F>,
    predictions: &ArrayView1<F>,
) -> F {
    let mut sum = SKCompensatedSum::new();
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        sum.add((*target - *prediction).abs());
    }
    sk_mean(sum.value(), targets.len())
}

/// Divide a total by a positive observation count.
fn sk_mean<F: SKFloat>(total: F, count: usize) -> F {
    let count = num_traits::cast::<usize, F>(count).expect("counts fit in the scalar");
    total / count
}
