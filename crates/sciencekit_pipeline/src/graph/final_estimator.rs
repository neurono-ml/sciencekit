//! Type-erased final supervised estimator and fitted final model adapters for
//! [`super::SKPipelineGraph`] (spec `pipeline-graph`).

use std::any::Any;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKError, SKFeatureBlock, SKFloat, SKRegressorPredictor, SKSupervisedFit,
    SKTargetView,
};

use crate::pipeline::sk_fit_final;

/// Type-erased fitted final supervised model stored by
/// [`super::SKPipelineGraph`]'s fitted graph model.
pub struct SKBoxedFinalModel<F: SKFloat> {
    inner: Arc<dyn Any + Send + Sync>,
    predictor: FinalPredictorKind<F>,
}

pub(super) enum FinalPredictorKind<F: SKFloat> {
    Classifier(Box<dyn ErasedClassifierModel<F>>),
    Regressor(Box<dyn ErasedRegressorModel<F>>),
    SupervisedOnly,
}

impl<F: SKFloat> SKBoxedFinalModel<F> {
    /// Borrow the underlying concrete fitted final model by type.
    pub fn downcast_reference<Model: 'static>(&self) -> Result<&Model, SKError> {
        self.inner.downcast_ref::<Model>().ok_or_else(|| {
            SKError::Conversion(
                "pipeline graph final model downcast requested a mismatched concrete model type"
                    .to_string(),
            )
        })
    }

    /// Whether the final model supports classifier prediction (`predict_labels`
    /// and `predict_probabilities`).
    pub fn is_classifier(&self) -> bool {
        matches!(self.predictor, FinalPredictorKind::Classifier(_))
    }

    /// Whether the final model supports regressor prediction (`predict`).
    pub fn is_regressor(&self) -> bool {
        matches!(self.predictor, FinalPredictorKind::Regressor(_))
    }

    #[allow(dead_code)]
    pub(super) fn predict_labels(&self, block: &SKFeatureBlock<F>) -> Result<Array1<i64>, SKError> {
        match &self.predictor {
            FinalPredictorKind::Classifier(classifier) => classifier.predict_labels_block(block),
            _ => Err(SKError::UnsupportedOperation {
                operation: "predict_labels",
                reason: "final estimator was not registered as a classifier".to_string(),
            }),
        }
    }

    #[allow(dead_code)]
    pub(super) fn predict_probabilities(
        &self,
        block: &SKFeatureBlock<F>,
    ) -> Result<Array2<F>, SKError> {
        match &self.predictor {
            FinalPredictorKind::Classifier(classifier) => {
                classifier.predict_probabilities_block(block)
            }
            _ => Err(SKError::UnsupportedOperation {
                operation: "predict_probabilities",
                reason: "final estimator was not registered as a classifier".to_string(),
            }),
        }
    }

    #[allow(dead_code)]
    pub(super) fn predict(&self, block: &SKFeatureBlock<F>) -> Result<Array1<F>, SKError> {
        match &self.predictor {
            FinalPredictorKind::Regressor(regressor) => regressor.predict_block(block),
            _ => Err(SKError::UnsupportedOperation {
                operation: "predict",
                reason: "final estimator was not registered as a regressor".to_string(),
            }),
        }
    }
}

pub(super) trait ErasedClassifierModel<F: SKFloat>: Send + Sync {
    fn predict_labels_block(&self, block: &SKFeatureBlock<F>) -> Result<Array1<i64>, SKError>;
    fn predict_probabilities_block(&self, block: &SKFeatureBlock<F>) -> Result<Array2<F>, SKError>;
}

impl<F, Model> ErasedClassifierModel<F> for Arc<Model>
where
    F: SKFloat,
    Model: SKClassifierPredictor<F> + Send + Sync + 'static,
    Model::Error: Into<SKError>,
{
    fn predict_labels_block(&self, block: &SKFeatureBlock<F>) -> Result<Array1<i64>, SKError> {
        self.as_ref().predict_labels(block).map_err(Into::into)
    }

    fn predict_probabilities_block(&self, block: &SKFeatureBlock<F>) -> Result<Array2<F>, SKError> {
        self.as_ref()
            .predict_probabilities(block)
            .map_err(Into::into)
    }
}

pub(super) trait ErasedRegressorModel<F: SKFloat>: Send + Sync {
    fn predict_block(&self, block: &SKFeatureBlock<F>) -> Result<Array1<F>, SKError>;
}

impl<F, Model> ErasedRegressorModel<F> for Arc<Model>
where
    F: SKFloat,
    Model: SKRegressorPredictor<F> + Send + Sync + 'static,
    Model::Error: Into<SKError>,
{
    fn predict_block(&self, block: &SKFeatureBlock<F>) -> Result<Array1<F>, SKError> {
        self.as_ref().predict(block).map_err(Into::into)
    }
}

pub(super) trait GraphFinalEstimator<F: SKFloat>: Send + Sync {
    fn fit_final_block(
        &self,
        block: &SKFeatureBlock<F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<SKBoxedFinalModel<F>, SKError>;
}

pub(super) struct ClassifierFinalAdapter<Estimator>(pub(super) Estimator);
pub(super) struct RegressorFinalAdapter<Estimator>(pub(super) Estimator);
pub(super) struct SupervisedFinalAdapter<Estimator>(pub(super) Estimator);

impl<F, Estimator> GraphFinalEstimator<F> for ClassifierFinalAdapter<Estimator>
where
    F: SKFloat,
    Estimator: SKSupervisedFit<F> + Send + Sync,
    Estimator::Error: Into<SKError>,
    Estimator::Model: SKClassifierPredictor<F> + Send + Sync + 'static,
    <Estimator::Model as SKClassifierPredictor<F>>::Error: Into<SKError>,
{
    fn fit_final_block(
        &self,
        block: &SKFeatureBlock<F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<SKBoxedFinalModel<F>, SKError> {
        let model = Arc::new(sk_fit_final(&self.0, block, targets)?);
        let predictor = FinalPredictorKind::Classifier(Box::new(Arc::clone(&model)));
        let inner: Arc<dyn Any + Send + Sync> = model;
        Ok(SKBoxedFinalModel { inner, predictor })
    }
}

impl<F, Estimator> GraphFinalEstimator<F> for RegressorFinalAdapter<Estimator>
where
    F: SKFloat,
    Estimator: SKSupervisedFit<F> + Send + Sync,
    Estimator::Error: Into<SKError>,
    Estimator::Model: SKRegressorPredictor<F> + Send + Sync + 'static,
    <Estimator::Model as SKRegressorPredictor<F>>::Error: Into<SKError>,
{
    fn fit_final_block(
        &self,
        block: &SKFeatureBlock<F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<SKBoxedFinalModel<F>, SKError> {
        let model = Arc::new(sk_fit_final(&self.0, block, targets)?);
        let predictor = FinalPredictorKind::Regressor(Box::new(Arc::clone(&model)));
        let inner: Arc<dyn Any + Send + Sync> = model;
        Ok(SKBoxedFinalModel { inner, predictor })
    }
}

impl<F, Estimator> GraphFinalEstimator<F> for SupervisedFinalAdapter<Estimator>
where
    F: SKFloat,
    Estimator: SKSupervisedFit<F> + Send + Sync,
    Estimator::Error: Into<SKError>,
    Estimator::Model: Send + Sync + 'static,
{
    fn fit_final_block(
        &self,
        block: &SKFeatureBlock<F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<SKBoxedFinalModel<F>, SKError> {
        let model = Arc::new(sk_fit_final(&self.0, block, targets)?);
        let inner: Arc<dyn Any + Send + Sync> = model;
        Ok(SKBoxedFinalModel {
            inner,
            predictor: FinalPredictorKind::SupervisedOnly,
        })
    }
}
