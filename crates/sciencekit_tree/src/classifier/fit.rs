//! Classifier fitting.

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKLabelTable, SKOperationAttributes, SKOperationKind,
    SKTargetView, sk_canonicalize_labels, sk_run_operation,
};

use crate::class_weight::sk_expand_class_weights;
use crate::design::{SKDesign, sk_to_csc};
use crate::growth::{SKGrowthConfig, sk_grow_tree};

use super::core_implementation::SKDecisionTreeClassifier;
use super::criterion::SKClassifierCriterion;
use super::model::SKDecisionTreeClassifierModel;

/// Fit the classifier: canonicalize the classes, expand the class weights and
/// grow the tree under the configured constraints.
pub(super) fn sk_fit_classifier<F: SKFloat>(
    estimator: &SKDecisionTreeClassifier<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKDecisionTreeClassifierModel<F>, SKError> {
    let (rows_count, columns_count) = sk_dimensions(&features)?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count,
        columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend(),
    };
    sk_run_operation(attributes, || {
        let (class_indices, classes) = sk_canonicalize_classes(&targets)?;
        if class_indices.len() != rows_count {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                1,
                class_indices.len(),
                1,
            ));
        }
        let number_of_classes = classes.number_of_classes();
        if number_of_classes < 2 {
            return Err(SKError::InvalidHyperparameter {
                name: "targets",
                reason: "classification requires at least two classes".to_string(),
            });
        }
        let sample_weights =
            sk_expand_class_weights(&estimator.class_weight, &class_indices, number_of_classes)?;
        let configuration = SKGrowthConfig {
            maximum_depth: estimator.maximum_depth,
            minimum_samples_split: estimator.minimum_samples_split,
            minimum_samples_leaf: estimator.minimum_samples_leaf,
            maximum_features: estimator.maximum_features.clone(),
            splitter: estimator.splitter,
            random_seed: estimator.random_seed,
        };
        match features {
            SKDataView::Dense(dense) => {
                dense
                    .iter()
                    .position(|value| !value.is_finite())
                    .map(|position| Err::<(), _>(SKError::NonFiniteValue { position }))
                    .transpose()?;
                let design = SKDesign::Dense(dense);
                sk_fit_design(
                    estimator,
                    &design,
                    &configuration,
                    &class_indices,
                    &sample_weights,
                    classes,
                    number_of_classes,
                )
            }
            SKDataView::Sparse(csr) => {
                let csc = sk_to_csc(&csr);
                let design = SKDesign::Sparse {
                    csr,
                    csc: csc.view(),
                };
                design.validate_finite()?;
                sk_fit_design(
                    estimator,
                    &design,
                    &configuration,
                    &class_indices,
                    &sample_weights,
                    classes,
                    number_of_classes,
                )
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            }),
        }
    })
}

/// Grow on a prepared design and assemble the fitted model.
#[allow(clippy::too_many_arguments)]
fn sk_fit_design<F: SKFloat>(
    estimator: &SKDecisionTreeClassifier<F>,
    design: &SKDesign<'_, F>,
    configuration: &SKGrowthConfig<F>,
    class_indices: &[usize],
    sample_weights: &[F],
    classes: SKLabelTable,
    number_of_classes: usize,
) -> Result<SKDecisionTreeClassifierModel<F>, SKError> {
    let criterion = SKClassifierCriterion {
        class_indices,
        sample_weights,
        number_of_classes,
        criterion: estimator.criterion,
    };
    let (tree, feature_importances) = sk_grow_tree(&criterion, design, configuration);
    Ok(SKDecisionTreeClassifierModel {
        tree,
        classes,
        number_of_classes,
        feature_importances,
        criterion: estimator.criterion,
        backend: estimator.backend(),
        execution_intent: estimator.execution_intent(),
    })
}

/// Canonicalize integer or nominal class labels, rejecting continuous targets.
fn sk_canonicalize_classes<F: SKFloat>(
    targets: &SKTargetView<'_, F>,
) -> Result<(Vec<usize>, SKLabelTable), SKError> {
    match targets {
        SKTargetView::Nominal(labels) => Ok(sk_canonicalize_labels(labels)),
        SKTargetView::Integer(values) => {
            let owned: Vec<String> = values.iter().map(|value| value.to_string()).collect();
            let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
            Ok(sk_canonicalize_labels(&borrowed))
        }
        SKTargetView::Continuous(_) => Err(SKError::UnsupportedRepresentation {
            representation: "continuous",
            suggestion: "pass integer or nominal class labels to a classifier",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "pass integer or nominal class labels to a classifier",
        }),
    }
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
