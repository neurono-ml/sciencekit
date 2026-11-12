//! Kernel catalogue configuration and `gamma` resolution rules
//! (spec `kernel-functions`).

use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::validation::{
    sk_feature_population_variance, sk_kernel_view_dimensions, sk_validate_kernel_finiteness,
};

/// Closed catalogue of Mercer and similarity kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKKernel {
    /// Linear inner product `⟨x, y⟩`.
    Linear,
    /// Polynomial kernel `(gamma · ⟨x, y⟩ + coef0)^degree`.
    Polynomial,
    /// Radial-basis-function kernel `exp(-gamma · ‖x − y‖²)`.
    #[default]
    RadialBasisFunction,
    /// Hyperbolic-tangent kernel `tanh(gamma · ⟨x, y⟩ + coef0)`.
    Sigmoid,
    /// Caller-supplied square kernel matrix.
    Precomputed,
}

/// Policy determining the scalar `gamma` of a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SKKernelGamma<F> {
    /// `1 / (n_features · Var(X))`.
    #[default]
    Scale,
    /// `1 / n_features`.
    Auto,
    /// An explicit non-negative scalar value.
    Value(F),
}

/// Resolved numeric parameters consumed by the kernel matrix functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKKernelParameters<F: SKFloat> {
    /// Selected kernel function.
    pub kernel: SKKernel,
    /// Resolved non-negative `gamma` scalar.
    pub gamma: F,
    /// Polynomial degree (strictly positive).
    pub degree: u32,
    /// Additive offset for polynomial and sigmoid kernels.
    pub coef0: F,
}

impl<F: SKFloat> SKKernelParameters<F> {
    /// Construct resolved kernel parameters.
    pub fn new(kernel: SKKernel, gamma: F, degree: u32, coef0: F) -> Self {
        Self {
            kernel,
            gamma,
            degree,
            coef0,
        }
    }
}

/// Validate static kernel hyperparameters independent of a feature matrix.
pub fn sk_validate_kernel_hyperparameters<F: SKFloat>(
    _kernel: SKKernel,
    gamma: SKKernelGamma<F>,
    degree: u32,
    coef0: F,
) -> Result<(), SKError> {
    if degree == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "degree",
            reason: "polynomial degree must be strictly positive".to_owned(),
        });
    }
    if let SKKernelGamma::Value(value) = gamma {
        if !value.is_finite() || value < F::zero() {
            return Err(SKError::InvalidHyperparameter {
                name: "gamma",
                reason: "explicit gamma must be finite and non-negative".to_owned(),
            });
        }
    }
    if !coef0.is_finite() {
        return Err(SKError::InvalidHyperparameter {
            name: "coef0",
            reason: "kernel offset coef0 must be finite".to_owned(),
        });
    }
    Ok(())
}

/// Resolve a [`SKKernelGamma`] policy against a training feature view.
pub fn sk_resolve_kernel_gamma<F: SKFloat>(
    gamma: SKKernelGamma<F>,
    features: SKDataView<'_, F>,
) -> Result<F, SKError> {
    match gamma {
        SKKernelGamma::Scale => {
            let (_, columns) = sk_kernel_view_dimensions(&features)?;
            let variance = sk_feature_population_variance(features)?;
            if variance <= F::zero() {
                return Err(SKError::InvalidHyperparameter {
                    name: "gamma",
                    reason: "feature population variance is zero; supply an explicit gamma or Auto"
                        .to_owned(),
                });
            }
            let denominator = F::from(columns).expect("feature count fits scalar") * variance;
            Ok(F::one() / denominator)
        }
        SKKernelGamma::Auto => {
            let (rows, columns) = sk_kernel_view_dimensions(&features)?;
            if rows == 0 || columns == 0 {
                return Err(SKError::EmptyInput {
                    operation: "kernel_gamma",
                });
            }
            sk_validate_kernel_finiteness(&features)?;
            Ok(F::one() / F::from(columns).expect("feature count fits scalar"))
        }
        SKKernelGamma::Value(value) => {
            if !value.is_finite() || value < F::zero() {
                return Err(SKError::InvalidHyperparameter {
                    name: "gamma",
                    reason: "explicit gamma must be finite and non-negative".to_owned(),
                });
            }
            Ok(value)
        }
    }
}

/// Validate and resolve a full kernel configuration against training features.
pub fn sk_resolve_kernel_parameters<F: SKFloat>(
    kernel: SKKernel,
    gamma: SKKernelGamma<F>,
    degree: u32,
    coef0: F,
    features: SKDataView<'_, F>,
) -> Result<SKKernelParameters<F>, SKError> {
    sk_validate_kernel_hyperparameters(kernel, gamma, degree, coef0)?;
    let resolved_gamma = match kernel {
        SKKernel::Linear | SKKernel::Precomputed => match gamma {
            SKKernelGamma::Value(value) => value,
            SKKernelGamma::Scale | SKKernelGamma::Auto => F::zero(),
        },
        SKKernel::Polynomial | SKKernel::RadialBasisFunction | SKKernel::Sigmoid => {
            sk_resolve_kernel_gamma(gamma, features)?
        }
    };
    Ok(SKKernelParameters::new(
        kernel,
        resolved_gamma,
        degree,
        coef0,
    ))
}
