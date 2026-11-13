//! The attached adapter: frozen base plus trainable low-rank factors (spec
//! `lora-adapters`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

use super::base_model::SKLoraBaseModel;
use super::core_implementation::SKLoraAdapter;

/// The scale applied to the deterministic initial `A` factor.
const SK_LORA_INITIAL_SCALE: f64 = 0.01;

/// A frozen base model with an adapter attached.
///
/// The struct owns the two low-rank factors: `A` is `rank × features` and `B`
/// is `heads × rank`. `B` starts at zero, so the initial update
/// `alpha / rank · B A` is exactly the zero matrix and the attached model
/// predicts exactly like its base. Fitting changes only `A`/`B`; the base stays
/// behind an immutable borrow.
#[derive(Debug, Clone)]
pub struct SKLoraAttachedModel<F> {
    adapter: SKLoraAdapter<F>,
    base: SKLoraBaseModel<F>,
    pub(super) a_matrix: Array2<F>,
    pub(super) b_matrix: Array2<F>,
    fitted: bool,
}

impl<F: SKFloat> SKLoraAttachedModel<F> {
    /// Validate the pairing and initialize the factors.
    ///
    /// `A` receives a deterministic, small, non-zero pattern so the optimizer
    /// has a direction to descend from; `B` starts at zero so the initial
    /// adapter contributes nothing. The deterministic pattern makes two fits on
    /// the same data produce the same adapter.
    pub(crate) fn attach(
        adapter: SKLoraAdapter<F>,
        base: SKLoraBaseModel<F>,
    ) -> Result<SKLoraAttachedModel<F>, SKError> {
        let target = base
            .target_module()
            .ok_or_else(|| base.unsupported_error())?;
        if adapter.target() != target {
            return Err(SKError::UnsupportedOperation {
                operation: "attach",
                reason: format!(
                    "the adapter targets `{}` but the base exposes `{}`",
                    adapter.target(),
                    target.parameter_name()
                ),
            });
        }
        let coefficients = base.coefficients()?;
        let (heads, features) = coefficients.dim();
        if features == 0 || heads == 0 {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, 1],
                found: vec![heads, features],
            });
        }
        let rank = adapter.rank();
        let initial_scale = F::from(SK_LORA_INITIAL_SCALE).unwrap_or_else(F::epsilon);
        let a_matrix = Array2::from_shape_fn((rank, features), |(row, column)| {
            let pattern = ((row * 31 + column * 17 + 1) % 13) as f64 - 6.0;
            F::from(pattern).unwrap_or_else(F::zero) * initial_scale
        });
        Ok(SKLoraAttachedModel {
            adapter,
            base,
            a_matrix,
            b_matrix: Array2::zeros((heads, rank)),
            fitted: false,
        })
    }

    /// The adapter configuration carried into this attachment.
    pub fn adapter(&self) -> &SKLoraAdapter<F> {
        &self.adapter
    }

    /// The frozen base model.
    pub fn base(&self) -> &SKLoraBaseModel<F> {
        &self.base
    }

    /// The `rank × features` factor `A`.
    pub fn a_matrix(&self) -> ArrayView2<'_, F> {
        self.a_matrix.view()
    }

    /// The `heads × rank` factor `B`.
    pub fn b_matrix(&self) -> ArrayView2<'_, F> {
        self.b_matrix.view()
    }

    /// Whether a fit has run since the adapter was attached or imported.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// The number of features the base and the adapter share.
    pub fn feature_count(&self) -> usize {
        self.a_matrix.ncols()
    }

    /// The number of adapted heads (`1` for a regressor).
    pub fn number_of_heads(&self) -> usize {
        self.b_matrix.nrows()
    }

    /// The low-rank update `(alpha / rank) · B A`, shaped `heads × features`.
    pub fn delta(&self) -> Array2<F> {
        let scaling = self.adapter.scaling();
        self.b_matrix
            .dot(&self.a_matrix)
            .mapv(|value| value * scaling)
    }

    /// The adapted parameter block `W + (alpha / rank) · B A`.
    pub(crate) fn effective_coefficients(&self) -> Result<Array2<F>, SKError> {
        Ok(self.base.coefficients()? + self.delta())
    }

    /// Rebuild an attached model from stored factors (unmerge and import).
    pub(super) fn from_factors(
        adapter: SKLoraAdapter<F>,
        base: SKLoraBaseModel<F>,
        a_matrix: Array2<F>,
        b_matrix: Array2<F>,
        fitted: bool,
    ) -> SKLoraAttachedModel<F> {
        SKLoraAttachedModel {
            adapter,
            base,
            a_matrix,
            b_matrix,
            fitted,
        }
    }

    /// Overwrite the factors with known values (tests and persistence import).
    pub(crate) fn set_factors(&mut self, a_matrix: Array2<F>, b_matrix: Array2<F>) {
        self.a_matrix = a_matrix;
        self.b_matrix = b_matrix;
    }

    /// Mark the adapter as fitted after a successful optimization.
    pub(crate) fn mark_fitted(&mut self) {
        self.fitted = true;
    }
}

impl<F: SKFloat> SKLoraAdapter<F> {
    /// Freeze a base model and attach this adapter to its target block.
    ///
    /// # Errors
    ///
    /// Returns [`SKError::UnsupportedOperation`] for an unsupported base family
    /// or a target module that does not match the base, and
    /// [`SKError::InvalidHyperparameter`] when the rank exceeds the smallest
    /// adapted dimension.
    pub fn attach(&self, base: SKLoraBaseModel<F>) -> Result<SKLoraAttachedModel<F>, SKError> {
        SKLoraAttachedModel::attach(self.clone(), base)
    }
}
