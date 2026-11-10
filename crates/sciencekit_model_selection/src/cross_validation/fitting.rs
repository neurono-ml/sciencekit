//! Fitting a gathered partition through the supervised fit contract.

use sciencekit_common::{
    SKDataView, SKError, SKFeatureBlock, SKFloat, SKSupervisedFit, SKTargetView,
};

use crate::SKTargetPartition;

/// Fits `estimator` on a gathered feature block and target partition.
pub(crate) fn sk_fit_on_partition<F, E>(
    estimator: &E,
    features: &SKFeatureBlock<F>,
    targets: &SKTargetPartition<F>,
) -> Result<E::Model, SKError>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    SKError: From<E::Error>,
{
    targets.with_view(|target_view| sk_fit_on_views(estimator, features.view(), target_view))
}

fn sk_fit_on_views<F, E>(
    estimator: &E,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<E::Model, SKError>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    SKError: From<E::Error>,
{
    match (features, targets) {
        (SKDataView::Dense(features), SKTargetView::Continuous(targets)) => {
            estimator.fit(features, targets).map_err(SKError::from)
        }
        (SKDataView::Dense(features), SKTargetView::Integer(targets)) => {
            estimator.fit(features, targets).map_err(SKError::from)
        }
        (SKDataView::Dense(features), SKTargetView::Nominal(targets)) => {
            estimator.fit(features, targets).map_err(SKError::from)
        }
        (SKDataView::Sparse(features), SKTargetView::Continuous(targets)) => {
            estimator.fit(features, targets).map_err(SKError::from)
        }
        (SKDataView::Sparse(features), SKTargetView::Integer(targets)) => {
            estimator.fit(features, targets).map_err(SKError::from)
        }
        (SKDataView::Sparse(features), SKTargetView::Nominal(targets)) => {
            estimator.fit(features, targets).map_err(SKError::from)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to dense or CSR and the targets to a known representation",
        }),
    }
}
