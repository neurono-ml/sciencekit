//! SMO dual variable and gradient state (spec `support-vector-machines`).

use sciencekit_common::SKFloat;

/// Dual state shared by C-SVC (`L = n`) and epsilon-SVR (`L = 2n`).
#[derive(Debug, Clone)]
pub struct SKSmoState<F: SKFloat> {
    /// Current dual variables `alpha_t ∈ [0, U_t]`.
    pub alphas: Vec<F>,
    /// Variable signs `y_t ∈ {-1, +1}`.
    pub signs: Vec<i8>,
    /// Per-variable box upper bounds `U_t`.
    pub upper_bounds: Vec<F>,
    /// Linear terms `p_t` of the dual objective.
    pub linear_terms: Vec<F>,
    /// Current dual gradients `G_t = (Q alpha)_t + p_t`.
    pub gradients: Vec<F>,
    /// Active working-set candidate indices.
    pub active_indices: Vec<usize>,
}

impl<F: SKFloat> SKSmoState<F> {
    /// Initialize a zero-alpha SMO state from signs, bounds and linear terms.
    pub fn new(signs: Vec<i8>, upper_bounds: Vec<F>, linear_terms: Vec<F>) -> Self {
        let count = signs.len();
        debug_assert_eq!(upper_bounds.len(), count);
        debug_assert_eq!(linear_terms.len(), count);
        let gradients = linear_terms.clone();
        Self {
            alphas: vec![F::zero(); count],
            signs,
            upper_bounds,
            linear_terms,
            gradients,
            active_indices: (0..count).collect(),
        }
    }

    /// Scalar sign `+1` or `-1` of variable `index`.
    #[inline]
    pub fn sign_scalar(&self, index: usize) -> F {
        if self.signs[index] > 0 {
            F::one()
        } else {
            -F::one()
        }
    }

    /// Whether variable `index` belongs to `I_up(alpha)`.
    #[inline]
    pub fn is_upper(&self, index: usize) -> bool {
        let alpha = self.alphas[index];
        let upper = self.upper_bounds[index];
        (self.signs[index] > 0 && alpha < upper) || (self.signs[index] < 0 && alpha > F::zero())
    }

    /// Whether variable `index` belongs to `I_low(alpha)`.
    #[inline]
    pub fn is_lower(&self, index: usize) -> bool {
        let alpha = self.alphas[index];
        let upper = self.upper_bounds[index];
        (self.signs[index] < 0 && alpha < upper) || (self.signs[index] > 0 && alpha > F::zero())
    }

    /// Whether variable `index` sits strictly inside `(0, U_index)`.
    #[inline]
    pub fn is_free(&self, index: usize) -> bool {
        let alpha = self.alphas[index];
        alpha > F::zero() && alpha < self.upper_bounds[index]
    }

    /// Compute `-y_t * G_t` for variable `index`.
    #[inline]
    pub fn negative_signed_gradient(&self, index: usize) -> F {
        if self.signs[index] > 0 {
            -self.gradients[index]
        } else {
            self.gradients[index]
        }
    }

    /// Apply a two-variable update `(first, second)` and update active gradients.
    #[allow(clippy::too_many_arguments)]
    pub fn update_pair(
        &mut self,
        first: usize,
        second: usize,
        new_first: F,
        new_second: F,
        kernel_row_first: &[F],
        kernel_row_second: &[F],
        row_modulus: usize,
    ) {
        let delta_first = new_first - self.alphas[first];
        let delta_second = new_second - self.alphas[second];
        self.alphas[first] = new_first;
        self.alphas[second] = new_second;

        let signed_delta_first = self.sign_scalar(first) * delta_first;
        let signed_delta_second = self.sign_scalar(second) * delta_second;
        if signed_delta_first == F::zero() && signed_delta_second == F::zero() {
            return;
        }
        let signs = &self.signs;
        let gradients = &mut self.gradients;
        self.active_indices.iter().for_each(|&index| {
            let row_index = index % row_modulus;
            let increment = signed_delta_first * kernel_row_first[row_index]
                + signed_delta_second * kernel_row_second[row_index];
            if signs[index] > 0 {
                gradients[index] = gradients[index] + increment;
            } else {
                gradients[index] = gradients[index] - increment;
            }
        });
    }
}
