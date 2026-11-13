//! Loss selection and validated configuration for [`SKLinearSVC`]
//! (spec `linear-support-vector-classifier`).
//!
//! [`SKLinearSVC`]: super::SKLinearSVC

use sciencekit_common::{SKBackendKind, SKBuilderState, SKError, SKFloat};

use crate::svc::SKClassWeight;

/// Loss function minimized by the linear support-vector classifier.
///
/// Both members correspond to the two L2-regularized linear SVC formulations
/// solved by LIBLINEAR: the classic L1 hinge loss and its squared variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SKLinearSVCLoss {
    /// L1 hinge loss, `max(0, 1 - y · f(x))`, with dual bound `U = C`.
    Hinge,
    /// L2 squared hinge loss, `max(0, 1 - y · f(x))²`, with unbounded dual and
    /// diagonal shift `1 / (2C)`.
    #[default]
    SquaredHinge,
}

impl SKLinearSVCLoss {
    /// Human-readable loss name, used in diagnostics and model accessors.
    pub fn name(self) -> &'static str {
        match self {
            SKLinearSVCLoss::Hinge => "hinge",
            SKLinearSVCLoss::SquaredHinge => "squared_hinge",
        }
    }
}

/// Translate a binding-level loss discriminator (`0` hinge, `1` squared hinge)
/// into a validated [`SKLinearSVCLoss`].
///
/// Language bindings receive untrusted integers; any value outside the known
/// discriminator set fails closed with the central
/// [`SKError::InvalidHyperparameter`] variant.
pub fn sk_linear_svc_loss_from_code(code: u8) -> Result<SKLinearSVCLoss, SKError> {
    match code {
        0 => Ok(SKLinearSVCLoss::Hinge),
        1 => Ok(SKLinearSVCLoss::SquaredHinge),
        _ => Err(SKError::InvalidHyperparameter {
            name: "loss",
            reason: format!(
                "unsupported linear SVC loss code `{code}`; use 0 (hinge) or 1 (squared hinge)"
            ),
        }),
    }
}

/// Internal validated configuration carried by [`SKLinearSVC`].
///
/// [`SKLinearSVC`]: super::SKLinearSVC
#[derive(Debug, Clone)]
pub(crate) struct SKLinearSVCConfiguration<F: SKFloat> {
    pub(crate) regularization_c: F,
    pub(crate) loss: SKLinearSVCLoss,
    pub(crate) class_weight: SKClassWeight<F>,
    pub(crate) include_intercept: bool,
    pub(crate) tolerance: F,
    pub(crate) maximum_number_of_iterations: usize,
    pub(crate) backend: SKBackendKind,
    pub(crate) builder_state: SKBuilderState,
}
