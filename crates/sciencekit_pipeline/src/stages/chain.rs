//! Left-nested tuple composition: every `then` appends to the tuple, and a
//! recursive pair implementation walks the chain without a runtime stage list.

use sciencekit_common::{SKError, SKFloat};

use crate::{SKFeatureBlock, SKPipelineTransformer};

/// A chain of configured stages whose type records their order.
///
/// The empty chain is `()`; each appended stage turns the chain into
/// `(previous_chain, stage)`. The recursive pair implementation fits and
/// transforms every stage in declaration order, threading one block through the
/// chain and emitting one observability span per stage.
pub trait SKConfiguredStages<F: SKFloat> {
    /// The fitted models, in the same order and the same nested-tuple shape.
    type FittedStages;

    /// Fit every stage on the current block, returning the fitted models and
    /// the block the last stage produced.
    fn fit_stages(
        &self,
        block: SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, SKFeatureBlock<F>), SKError>;

    /// Transform a block through every fitted stage in order.
    fn transform_stages(
        &self,
        fitted: &Self::FittedStages,
        block: SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError>;

    /// The number of configured stages.
    fn number_of_stages(&self) -> usize;
}

impl<F: SKFloat> SKConfiguredStages<F> for () {
    type FittedStages = ();

    fn fit_stages(
        &self,
        block: SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, SKFeatureBlock<F>), SKError> {
        Ok(((), block))
    }

    fn transform_stages(
        &self,
        _fitted: &Self::FittedStages,
        block: SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError> {
        Ok(block)
    }

    fn number_of_stages(&self) -> usize {
        0
    }
}

impl<F, Previous, Last> SKConfiguredStages<F> for (Previous, Last)
where
    F: SKFloat,
    Previous: SKConfiguredStages<F>,
    Last: SKPipelineTransformer<F>,
{
    type FittedStages = (Previous::FittedStages, Last::Fitted);

    fn fit_stages(
        &self,
        block: SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, SKFeatureBlock<F>), SKError> {
        let (previous_fitted, previous_output) = self.0.fit_stages(block)?;
        let stage_span = tracing::info_span!(
            "pipeline_stage",
            index = self.0.number_of_stages(),
            operation = "fit"
        );
        let _guard = stage_span.enter();
        let fitted = self.1.fit_stage(&previous_output)?;
        let output = self.1.transform_stage(&fitted, &previous_output)?;
        Ok(((previous_fitted, fitted), output))
    }

    fn transform_stages(
        &self,
        fitted: &Self::FittedStages,
        block: SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError> {
        let previous_output = self.0.transform_stages(&fitted.0, block)?;
        let stage_span = tracing::info_span!(
            "pipeline_stage",
            index = self.0.number_of_stages(),
            operation = "transform"
        );
        let _guard = stage_span.enter();
        self.1.transform_stage(&fitted.1, &previous_output)
    }

    fn number_of_stages(&self) -> usize {
        self.0.number_of_stages() + 1
    }
}
