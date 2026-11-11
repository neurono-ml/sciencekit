//! Dynamic boxed fallback (`Vec<Box<dyn SKBlockTransformer<F>>>`) for runtime
//! [`super::SKFeatureUnion`] construction (spec `feature-union`).

use std::any::Any;

use sciencekit_common::{SKError, SKFloat};

use crate::{SKFeatureBlock, SKPipelineTransformer};

use super::builder::{SKFeatureUnion, SKFeatureUnionBuilder};
use super::errors::{SKFeatureUnionMemberError, sk_validate_probe_output};
use super::stages::{SKFeatureUnionArity, SKFeatureUnionStages};

/// Type-erased fitted state of one dynamic [`SKBlockTransformer`] member.
pub struct SKBoxedFittedStage {
    inner: Box<dyn Any + Send + Sync>,
}

impl SKBoxedFittedStage {
    /// Wrap a concrete fitted stage model for dynamic storage.
    pub fn new<Fitted: Send + Sync + 'static>(fitted: Fitted) -> Self {
        Self {
            inner: Box::new(fitted),
        }
    }

    /// Borrow the underlying concrete fitted stage model.
    pub fn downcast_ref<Fitted: 'static>(&self) -> Result<&Fitted, SKError> {
        self.inner.downcast_ref::<Fitted>().ok_or_else(|| {
            SKError::Conversion(
                "boxed feature union member received a mismatched fitted stage type".to_string(),
            )
        })
    }
}

/// Object-safe block transformer contract for dynamic [`SKFeatureUnion`] lists.
///
/// Every `Send + Sync` [`SKPipelineTransformer`] whose fitted model is
/// `Send + Sync + 'static` implements this trait automatically, so callers can
/// collect `Vec<Box<dyn SKBlockTransformer<F>>>` at runtime.
pub trait SKBlockTransformer<F: SKFloat>: Send + Sync {
    /// Fit this member on `block` and return its type-erased fitted state.
    fn fit_block(&self, block: &SKFeatureBlock<F>) -> Result<SKBoxedFittedStage, SKError>;

    /// Transform `block` using `fitted`, which came from [`Self::fit_block`].
    fn transform_block(
        &self,
        fitted: &SKBoxedFittedStage,
        block: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError>;
}

impl<F, Stage> SKBlockTransformer<F> for Stage
where
    F: SKFloat,
    Stage: SKPipelineTransformer<F> + Send + Sync,
    Stage::Fitted: Send + Sync + 'static,
{
    fn fit_block(&self, block: &SKFeatureBlock<F>) -> Result<SKBoxedFittedStage, SKError> {
        let fitted = self.fit_stage(block)?;
        Ok(SKBoxedFittedStage::new(fitted))
    }

    fn transform_block(
        &self,
        fitted: &SKBoxedFittedStage,
        block: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError> {
        let concrete = fitted.downcast_ref::<Stage::Fitted>()?;
        self.transform_stage(concrete, block)
    }
}

impl<F: SKFloat> SKFeatureUnionArity for Vec<Box<dyn SKBlockTransformer<F>>> {
    fn number_of_members(&self) -> usize {
        self.len()
    }
}

impl<F: SKFloat> SKFeatureUnionStages<F> for Vec<Box<dyn SKBlockTransformer<F>>> {
    type FittedStages = Vec<SKBoxedFittedStage>;

    fn fit_stages_checked(
        &self,
        block: &SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, Vec<usize>), SKFeatureUnionMemberError> {
        let expected_rows = block.number_of_rows();
        let pairs: Vec<(SKBoxedFittedStage, usize)> = self
            .iter()
            .enumerate()
            .map(|(member_index, stage)| {
                let span = tracing::info_span!(
                    "feature_union_member",
                    member_index = member_index,
                    operation = "fit"
                );
                let _guard = span.enter();
                let fitted = stage
                    .fit_block(block)
                    .map_err(|error| SKFeatureUnionMemberError::member(member_index, error))?;
                let probe = stage
                    .transform_block(&fitted, block)
                    .map_err(|error| SKFeatureUnionMemberError::member(member_index, error))?;
                let width = sk_validate_probe_output(member_index, &probe, expected_rows)?;
                Ok((fitted, width))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let (fitted_stages, widths) = pairs.into_iter().unzip();
        Ok((fitted_stages, widths))
    }

    fn transform_stages_checked(
        &self,
        fitted: &Self::FittedStages,
        block: &SKFeatureBlock<F>,
    ) -> Result<Vec<SKFeatureBlock<F>>, SKFeatureUnionMemberError> {
        self.iter()
            .zip(fitted.iter())
            .enumerate()
            .map(|(member_index, (stage, member_fitted))| {
                let span = tracing::info_span!(
                    "feature_union_member",
                    member_index = member_index,
                    operation = "transform"
                );
                let _guard = span.enter();
                stage
                    .transform_block(member_fitted, block)
                    .map_err(|error| SKFeatureUnionMemberError::member(member_index, error))
            })
            .collect()
    }
}

impl SKFeatureUnion<()> {
    /// Start a dynamically-sized feature-union builder backed by
    /// `Vec<Box<dyn SKBlockTransformer<F>>>`.
    pub fn boxed<F: SKFloat>() -> SKFeatureUnionBuilder<Vec<Box<dyn SKBlockTransformer<F>>>> {
        SKFeatureUnionBuilder::new(Vec::new())
    }
}

impl<F: SKFloat> SKFeatureUnionBuilder<Vec<Box<dyn SKBlockTransformer<F>>>> {
    /// Push a concrete transformer stage onto the dynamic boxed member list.
    pub fn with_boxed_stage<Stage>(mut self, stage: Stage) -> Self
    where
        Stage: SKBlockTransformer<F> + 'static,
    {
        self.stages.push(Box::new(stage));
        self
    }
}
