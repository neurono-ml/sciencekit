//! The logistic solve orchestration: the multi-class strategy dispatch and the
//! solver selection (spec `logistic-solver`, Decisions 2 and 4).

use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;
use sciencekit_common::{SKError, SKFloat};

use super::binary::sk_solve_binary_head;
use super::binary_saga::sk_solve_binary_head_saga;
use super::lbfgs::SKLbfgsConfig;
use super::liblinear_dual::sk_solve_binary_head_liblinear_l2;
use super::liblinear_primal::sk_solve_binary_head_liblinear_l1;
use super::multinomial::sk_solve_multinomial;
use super::multinomial_saga::sk_solve_multinomial_saga;
use super::solution::SKLogisticSolution;
use crate::logistic::config::{SKLogisticConfig, SKLogisticPenalty, SKLogisticSolver};
use crate::logistic::design::SKLogisticDesign;
use crate::logistic::resolution::SKResolvedMultiClass;

/// One solved head: the feature weights, the intercept and the iterations spent.
type SKHeadResult<F> = Result<(Array1<F>, F, usize), SKError>;

/// Solve the configured logistic objective for the resolved strategy.
///
/// An optional previous solution seeds the optimizer when warm start is
/// enabled.
pub(crate) fn sk_solve_logistic<F, D>(
    design: &D,
    labels: &[usize],
    sample_weights: &[F],
    number_of_classes: usize,
    strategy: SKResolvedMultiClass,
    config: &SKLogisticConfig<F>,
    initial: Option<&SKLogisticSolution<F>>,
) -> Result<SKLogisticSolution<F>, SKError>
where
    F: SKFloat,
    D: SKLogisticDesign<F> + Sync,
{
    match strategy {
        SKResolvedMultiClass::Binary => {
            let binary_labels = sk_binary_labels(labels, None);
            let seed = sk_binary_initial(initial, 0, design.columns_count(), config);
            let (coefficients, intercept, _) =
                sk_solve_binary(design, &binary_labels, sample_weights, seed, config)?;
            Ok(SKLogisticSolution {
                coefficients: coefficients.insert_axis(Axis(0)),
                intercepts: Array1::from_vec(vec![intercept]),
            })
        }
        SKResolvedMultiClass::OneVersusRest => {
            let heads: Vec<SKHeadResult<F>> = (0..number_of_classes)
                .into_par_iter()
                .map(|class| {
                    let binary_labels = sk_binary_labels(labels, Some(class));
                    let seed = sk_binary_initial(initial, class, design.columns_count(), config);
                    sk_solve_binary(design, &binary_labels, sample_weights, seed, config)
                })
                .collect();
            let mut coefficients = Array2::<F>::zeros((number_of_classes, design.columns_count()));
            let mut intercepts = Array1::<F>::zeros(number_of_classes);
            for (class, head) in heads.into_iter().enumerate() {
                let (weights, intercept, _) = head?;
                coefficients.row_mut(class).assign(&weights);
                intercepts[class] = intercept;
            }
            Ok(SKLogisticSolution {
                coefficients,
                intercepts,
            })
        }
        SKResolvedMultiClass::Multinomial => {
            let seed =
                sk_multinomial_initial(initial, number_of_classes, design.columns_count(), config);
            let (coefficients, intercepts, _) = sk_solve_multinomial_strategy(
                design,
                labels,
                sample_weights,
                number_of_classes,
                seed,
                config,
            )?;
            Ok(SKLogisticSolution {
                coefficients,
                intercepts,
            })
        }
    }
}

/// Fit one binary head with the configured solver.
fn sk_solve_binary<F, D>(
    design: &D,
    labels: &[F],
    sample_weights: &[F],
    initial: Option<Array1<F>>,
    config: &SKLogisticConfig<F>,
) -> Result<(Array1<F>, F, usize), SKError>
where
    F: SKFloat,
    D: SKLogisticDesign<F> + Sync,
{
    match config.solver {
        SKLogisticSolver::Lbfgs => {
            let solution = sk_solve_binary_head(
                design,
                labels,
                sample_weights,
                config.alpha,
                config.penalty == SKLogisticPenalty::L2,
                config.include_intercept,
                initial,
                &sk_lbfgs_config(config),
            )?;
            Ok((
                solution.coefficients,
                solution.intercept,
                solution.iterations,
            ))
        }
        SKLogisticSolver::Saga => {
            sk_solve_binary_head_saga(design, labels, sample_weights, initial, config)
        }
        SKLogisticSolver::Liblinear => match config.penalty {
            SKLogisticPenalty::L1 => {
                sk_solve_binary_head_liblinear_l1(design, labels, sample_weights, config)
            }
            _ => sk_solve_binary_head_liblinear_l2(design, labels, sample_weights, config),
        },
    }
}

/// Fit the joint multinomial head set with the configured solver.
fn sk_solve_multinomial_strategy<F, D>(
    design: &D,
    labels: &[usize],
    sample_weights: &[F],
    number_of_classes: usize,
    initial: Option<Array1<F>>,
    config: &SKLogisticConfig<F>,
) -> Result<(Array2<F>, Array1<F>, usize), SKError>
where
    F: SKFloat,
    D: SKLogisticDesign<F> + Sync,
{
    match config.solver {
        SKLogisticSolver::Saga => sk_solve_multinomial_saga(
            design,
            labels,
            sample_weights,
            number_of_classes,
            initial,
            config,
        ),
        SKLogisticSolver::Lbfgs => {
            let solution = sk_solve_multinomial(
                design,
                labels,
                sample_weights,
                number_of_classes,
                config.alpha,
                config.penalty == SKLogisticPenalty::L2,
                config.include_intercept,
                initial,
                &sk_lbfgs_config(config),
            )?;
            Ok((
                solution.coefficients,
                solution.intercepts,
                solution.iterations,
            ))
        }
        SKLogisticSolver::Liblinear => Err(SKError::InvalidHyperparameter {
            name: "multi_class",
            reason: "the Liblinear solver supports only one-vs-rest".to_string(),
        }),
    }
}

/// The shared L-BFGS controls derived from the objective configuration.
fn sk_lbfgs_config<F: SKFloat>(config: &SKLogisticConfig<F>) -> SKLbfgsConfig<F> {
    SKLbfgsConfig {
        tolerance: config.tolerance,
        maximum_number_of_iterations: config.maximum_number_of_iterations,
        history_size: 10,
    }
}

/// The `0.0`/`1.0` binary labels, optionally one-vs-rest against `class`.
fn sk_binary_labels<F: SKFloat>(labels: &[usize], class: Option<usize>) -> Vec<F> {
    labels
        .iter()
        .map(|&label| {
            let positive = match class {
                Some(target) => label == target,
                None => label == 1,
            };
            if positive { F::one() } else { F::zero() }
        })
        .collect()
}

/// The flattened `[w, b]` seed for one binary head, when a solution is given.
fn sk_binary_initial<F: SKFloat>(
    initial: Option<&SKLogisticSolution<F>>,
    head: usize,
    columns: usize,
    config: &SKLogisticConfig<F>,
) -> Option<Array1<F>> {
    let solution = initial?;
    if head >= solution.coefficients.nrows() {
        return None;
    }
    let dimension = columns + usize::from(config.include_intercept);
    let mut point = Array1::<F>::zeros(dimension);
    point
        .slice_mut(s![..columns])
        .assign(&solution.coefficients.row(head));
    if config.include_intercept {
        point[columns] = solution.intercepts[head];
    }
    Some(point)
}

/// The flattened `[W, b]` seed for the multinomial head set, when given.
fn sk_multinomial_initial<F: SKFloat>(
    initial: Option<&SKLogisticSolution<F>>,
    number_of_classes: usize,
    columns: usize,
    config: &SKLogisticConfig<F>,
) -> Option<Array1<F>> {
    let solution = initial?;
    if solution.coefficients.nrows() != number_of_classes {
        return None;
    }
    let dimension =
        number_of_classes * columns + usize::from(config.include_intercept) * number_of_classes;
    let mut point = Array1::<F>::zeros(dimension);
    for head in 0..number_of_classes {
        point
            .slice_mut(s![head * columns..(head + 1) * columns])
            .assign(&solution.coefficients.row(head));
        if config.include_intercept {
            point[number_of_classes * columns + head] = solution.intercepts[head];
        }
    }
    Some(point)
}
