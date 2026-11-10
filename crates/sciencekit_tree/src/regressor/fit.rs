//! Regressor fitting.

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKTargetView,
    sk_run_operation,
};

use crate::design::{SKDesign, sk_to_csc};
use crate::growth::{SKGrowthConfig, sk_grow_tree};

use super::core_implementation::SKDecisionTreeRegressor;
use super::criterion::SKRegressorCriterion;
use super::model::SKDecisionTreeRegressorModel;

/// Fit the regressor: validate the continuous responses and grow the tree.
pub(super) fn sk_fit_regressor<F: SKFloat>(
    estimator: &SKDecisionTreeRegressor<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKDecisionTreeRegressorModel<F>, SKError> {
    let (rows_count, columns_count) = sk_dimensions(&features)?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count,
        columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend(),
    };
    sk_run_operation(attributes, || {
        let responses = targets.as_continuous()?;
        if responses.len() != rows_count {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                1,
                responses.len(),
                1,
            ));
        }
        if let Some(position) = responses.iter().position(|value| !value.is_finite()) {
            return Err(SKError::NonFiniteValue { position });
        }
        let configuration = SKGrowthConfig {
            maximum_depth: estimator.maximum_depth,
            minimum_samples_split: estimator.minimum_samples_split,
            minimum_samples_leaf: estimator.minimum_samples_leaf,
            maximum_features: estimator.maximum_features.clone(),
            splitter: estimator.splitter,
            random_seed: estimator.random_seed,
        };
        let responses_values: Vec<F> = responses.iter().copied().collect();
        match features {
            SKDataView::Dense(dense) => {
                if let Some(position) = dense.iter().position(|value| !value.is_finite()) {
                    return Err(SKError::NonFiniteValue { position });
                }
                let design = SKDesign::Dense(dense);
                sk_fit_design(estimator, &design, &configuration, &responses_values)
            }
            SKDataView::Sparse(csr) => {
                let csc = sk_to_csc(&csr);
                let design = SKDesign::Sparse {
                    csr,
                    csc: csc.view(),
                };
                design.validate_finite()?;
                sk_fit_design(estimator, &design, &configuration, &responses_values)
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            }),
        }
    })
}

/// Grow on a prepared design and assemble the fitted model.
fn sk_fit_design<F: SKFloat>(
    estimator: &SKDecisionTreeRegressor<F>,
    design: &SKDesign<'_, F>,
    configuration: &SKGrowthConfig<F>,
    responses: &[F],
) -> Result<SKDecisionTreeRegressorModel<F>, SKError> {
    let criterion = SKRegressorCriterion {
        responses,
        criterion: estimator.criterion,
    };
    let (tree, feature_importances) = sk_grow_tree(&criterion, design, configuration);
    Ok(SKDecisionTreeRegressorModel {
        tree,
        feature_importances,
        criterion: estimator.criterion,
        backend: estimator.backend(),
        execution_intent: estimator.execution_intent(),
    })
}

/// Validate the raw shape and return `(rows, columns)`.
fn sk_dimensions<F>(features: &SKDataView<'_, F>) -> Result<(usize, usize), SKError> {
    let (rows, columns) = match features {
        SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    };
    if rows == 0 {
        return Err(SKError::EmptyInput { operation: "fit" });
    }
    if columns == 0 {
        return Err(SKError::shape_mismatch_2d(rows, 1, rows, 0));
    }
    Ok((rows, columns))
}
