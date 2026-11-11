//! Platt scaling sigmoid calibration (Lin, Lin & Weng 2007)
//! (spec `support-vector-machines`).

use sciencekit_common::SKFloat;

/// Fitted Platt scaling parameters `P(y = +1 | f) = 1 / (1 + exp(slope * f + intercept))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKPlattParameters<F: SKFloat> {
    /// Sigmoid slope `A` (negative when larger margins indicate `+1`).
    pub slope: F,
    /// Sigmoid intercept `B`.
    pub intercept: F,
}

impl<F: SKFloat> SKPlattParameters<F> {
    /// Construct Platt parameters from explicit `(slope, intercept)`.
    pub fn new(slope: F, intercept: F) -> Self {
        Self { slope, intercept }
    }

    /// Evaluate calibrated `P(y = +1 | decision_value) ∈ [0, 1]` without overflow.
    pub fn predict_positive_probability(&self, decision_value: F) -> F {
        let exponent_argument = self.slope * decision_value + self.intercept;
        let probability = if exponent_argument >= F::zero() {
            let neg_exp = (-exponent_argument).exp();
            neg_exp / (F::one() + neg_exp)
        } else {
            F::one() / (F::one() + exponent_argument.exp())
        };
        probability.max(F::zero()).min(F::one())
    }
}

fn sk_platt_objective(margins: &[f64], targets: &[f64], slope: f64, intercept: f64) -> f64 {
    margins
        .iter()
        .zip(targets.iter())
        .fold(0.0_f64, |total, (&margin, &target)| {
            let q = margin * slope + intercept;
            let term = if q >= 0.0 {
                target * q + (1.0 + (-q).exp()).ln()
            } else {
                (target - 1.0) * q + (1.0 + q.exp()).ln()
            };
            total + term
        })
}

/// Fit Platt sigmoid parameters by regularized Newton maximum likelihood
/// (Lin, Lin & Weng 2007, Algorithm 1).
pub fn sk_fit_platt_sigmoid<F: SKFloat>(
    decision_values: &[F],
    signs: &[i8],
) -> SKPlattParameters<F> {
    let margins: Vec<f64> = decision_values
        .iter()
        .map(|value| value.to_f64().unwrap_or(0.0))
        .collect();
    let positive_count = signs.iter().filter(|&&sign| sign > 0).count() as f64;
    let negative_count = signs.iter().filter(|&&sign| sign < 0).count() as f64;

    let target_positive = (positive_count + 1.0) / (positive_count + 2.0);
    let target_negative = 1.0 / (negative_count + 2.0);
    let targets: Vec<f64> = signs
        .iter()
        .map(|&sign| if sign > 0 { target_positive } else { target_negative })
        .collect();

    let mut slope = 0.0_f64;
    let mut intercept = ((negative_count + 1.0) / (positive_count + 1.0)).ln();
    let mut current_loss = sk_platt_objective(&margins, &targets, slope, intercept);

    let damping = 1e-12_f64;
    let min_step = 1e-10_f64;
    let stopping_tolerance = 1e-5_f64;

    for _ in 0..100 {
        let mut h_aa = damping;
        let mut h_bb = damping;
        let mut h_ab = 0.0_f64;
        let mut g_a = 0.0_f64;
        let mut g_b = 0.0_f64;

        for (&margin, &target) in margins.iter().zip(targets.iter()) {
            let q = margin * slope + intercept;
            let (p, one_minus_p) = if q >= 0.0 {
                let exp_neg = (-q).exp();
                let denom = 1.0 + exp_neg;
                (exp_neg / denom, 1.0 / denom)
            } else {
                let exp_pos = q.exp();
                let denom = 1.0 + exp_pos;
                (1.0 / denom, exp_pos / denom)
            };
            let curvature = p * one_minus_p;
            let residual = target - p;
            g_a += margin * residual;
            g_b += residual;
            h_aa += margin * margin * curvature;
            h_bb += curvature;
            h_ab += margin * curvature;
        }

        if g_a.abs() < stopping_tolerance && g_b.abs() < stopping_tolerance {
            break;
        }

        let determinant = h_aa * h_bb - h_ab * h_ab;
        if determinant.abs() <= 1e-24 {
            break;
        }
        let step_a = -(h_bb * g_a - h_ab * g_b) / determinant;
        let step_b = -(-h_ab * g_a + h_aa * g_b) / determinant;
        let directional_derivative = g_a * step_a + g_b * step_b;

        let mut step_size = 1.0_f64;
        while step_size >= min_step {
            let cand_a = slope + step_size * step_a;
            let cand_b = intercept + step_size * step_b;
            let cand_loss = sk_platt_objective(&margins, &targets, cand_a, cand_b);
            if cand_loss < current_loss + 1e-4 * step_size * directional_derivative {
                slope = cand_a;
                intercept = cand_b;
                current_loss = cand_loss;
                break;
            }
            step_size *= 0.5;
        }
        if step_size < min_step {
            break;
        }
    }

    SKPlattParameters::new(
        F::from(slope).expect("Platt slope fits SKFloat"),
        F::from(intercept).expect("Platt intercept fits SKFloat"),
    )
}
