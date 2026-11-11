//! Pipeline-stage integration: [`super::SKFeatureUnion`] inside
//! [`crate::SKPipeline`], and non-empty [`crate::SKPipelineBuilder`] chains as
//! branches inside [`super::SKFeatureUnion`] (spec `feature-union`).

use sciencekit_common::{SKError, SKOperationAttributes, SKOperationKind, sk_run_operation};

use crate::{SKConfiguredStages, SKFeatureBlock, SKPipelineBuilder, SKPipelineTransformer};

use super::builder::SKFeatureUnion;
use super::model::SKFeatureUnionModel;
use super::stages::SKFeatureUnionStages;

macro_rules! impl_pipeline_stage_for_float {
    ($scalar:ty) => {
        impl<Stages> SKPipelineTransformer<$scalar> for SKFeatureUnion<Stages>
        where
            Stages: SKFeatureUnionStages<$scalar>,
        {
            type Fitted = SKFeatureUnionModel<$scalar, Stages>;

            fn fit_stage(&self, block: &SKFeatureBlock<$scalar>) -> Result<Self::Fitted, SKError> {
                self.fit(block)
            }

            fn transform_stage(
                &self,
                fitted: &Self::Fitted,
                block: &SKFeatureBlock<$scalar>,
            ) -> Result<SKFeatureBlock<$scalar>, SKError> {
                fitted.transform(block)
            }
        }

        impl<Previous, Last> SKPipelineTransformer<$scalar>
            for SKPipelineBuilder<$scalar, (Previous, Last)>
        where
            Previous: SKConfiguredStages<$scalar>,
            Last: SKPipelineTransformer<$scalar>,
        {
            type Fitted = <(Previous, Last) as SKConfiguredStages<$scalar>>::FittedStages;

            fn fit_stage(&self, block: &SKFeatureBlock<$scalar>) -> Result<Self::Fitted, SKError> {
                let attributes = SKOperationAttributes {
                    operation: SKOperationKind::Fit,
                    rows_count: block.number_of_rows(),
                    columns_count: block.number_of_columns(),
                    execution_mode: self.execution_intent,
                    backend: self.backend,
                };
                sk_run_operation(attributes, || {
                    let (fitted, _) = self.stages.fit_stages(block.clone())?;
                    Ok(fitted)
                })
            }

            fn transform_stage(
                &self,
                fitted: &Self::Fitted,
                block: &SKFeatureBlock<$scalar>,
            ) -> Result<SKFeatureBlock<$scalar>, SKError> {
                let attributes = SKOperationAttributes {
                    operation: SKOperationKind::Transform,
                    rows_count: block.number_of_rows(),
                    columns_count: block.number_of_columns(),
                    execution_mode: self.execution_intent,
                    backend: self.backend,
                };
                sk_run_operation(attributes, || {
                    self.stages.transform_stages(fitted, block.clone())
                })
            }
        }
    };
}

impl_pipeline_stage_for_float!(f32);
impl_pipeline_stage_for_float!(f64);
