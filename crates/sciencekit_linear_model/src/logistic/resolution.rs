//! The penalty/solver and multi-class resolution of the logistic substrate
//! (spec `logistic-solver`).

use sciencekit_common::SKError;

use super::config::{SKLogisticPenalty, SKLogisticSolver, SKMultiClass};

/// The concrete fitting strategy a configured multi-class choice resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SKResolvedMultiClass {
    /// A single binary head (two classes).
    Binary,
    /// One binary head per class.
    OneVersusRest,
    /// A single joint softmax head set.
    Multinomial,
}

/// Reject a solver/penalty pair the solver cannot solve, following
/// scikit-learn's matrix: `Lbfgs` supports `L2`/`None`, `Liblinear` supports
/// `L1`/`L2`, `Saga` supports every penalty.
pub(crate) fn sk_validate_solver_penalty(
    solver: SKLogisticSolver,
    penalty: SKLogisticPenalty,
) -> Result<(), SKError> {
    let supported = match solver {
        SKLogisticSolver::Lbfgs => {
            matches!(penalty, SKLogisticPenalty::L2 | SKLogisticPenalty::None)
        }
        SKLogisticSolver::Liblinear => {
            matches!(penalty, SKLogisticPenalty::L1 | SKLogisticPenalty::L2)
        }
        SKLogisticSolver::Saga => true,
    };
    if supported {
        Ok(())
    } else {
        Err(SKError::InvalidHyperparameter {
            name: "penalty",
            reason: format!(
                "the {solver:?} solver does not support the {penalty:?} penalty; \
                 use a solver from its supported set"
            ),
        })
    }
}

/// Resolve the configured multi-class choice against the solver and the class
/// count.
///
/// A single class is rejected; `Liblinear` with `Multinomial` is rejected;
/// two classes always fit a single binary head; `Auto` resolves to
/// one-vs-rest for `Liblinear` and to multinomial for the smooth solvers on
/// more than two classes.
pub(crate) fn sk_resolve_multi_class(
    multi_class: SKMultiClass,
    solver: SKLogisticSolver,
    number_of_classes: usize,
) -> Result<SKResolvedMultiClass, SKError> {
    if number_of_classes < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "targets",
            reason: "at least two classes are required for a classification fit".to_string(),
        });
    }
    if solver == SKLogisticSolver::Liblinear && multi_class == SKMultiClass::Multinomial {
        return Err(SKError::InvalidHyperparameter {
            name: "multi_class",
            reason: "the Liblinear solver supports only one-vs-rest".to_string(),
        });
    }
    if number_of_classes == 2 {
        return Ok(SKResolvedMultiClass::Binary);
    }
    Ok(match multi_class {
        SKMultiClass::OneVersusRest => SKResolvedMultiClass::OneVersusRest,
        SKMultiClass::Multinomial => SKResolvedMultiClass::Multinomial,
        SKMultiClass::Auto => match solver {
            SKLogisticSolver::Liblinear => SKResolvedMultiClass::OneVersusRest,
            SKLogisticSolver::Lbfgs | SKLogisticSolver::Saga => SKResolvedMultiClass::Multinomial,
        },
    })
}
