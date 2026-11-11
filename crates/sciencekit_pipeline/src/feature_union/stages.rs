//! Tuple-based member composition for [`super::SKFeatureUnion`] (spec
//! `feature-union`).

use sciencekit_common::{SKError, SKFloat};

use crate::{SKFeatureBlock, SKPipelineTransformer};

/// Scalar-independent member count of a feature-union stage collection.
pub trait SKFeatureUnionArity {
    /// The number of configured member transformers in the union.
    fn number_of_members(&self) -> usize;
}

/// Heterogeneous tuple of member transformers fitted on the same input block.
///
/// Implemented for tuples from one up to eight [`SKPipelineTransformer`] stages
/// (plus the empty tuple `()`, rejected at builder validation) so each member
/// preserves its concrete static type without boxing.
pub trait SKFeatureUnionStages<F: SKFloat>: SKFeatureUnionArity {
    /// The tuple (or list) of fitted models corresponding 1-to-1 to the members.
    type FittedStages;

    /// Fit every member on `block`, returning the fitted models and each
    /// member's output width (after checking row alignment against `block`).
    fn fit_stages(
        &self,
        block: &SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, Vec<usize>), SKError>;

    /// Transform `block` with every fitted member in declaration order.
    fn transform_stages(
        &self,
        fitted: &Self::FittedStages,
        block: &SKFeatureBlock<F>,
    ) -> Result<Vec<SKFeatureBlock<F>>, SKError>;
}

impl SKFeatureUnionArity for () {
    fn number_of_members(&self) -> usize {
        0
    }
}

impl<F: SKFloat> SKFeatureUnionStages<F> for () {
    type FittedStages = ();

    fn fit_stages(
        &self,
        _block: &SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, Vec<usize>), SKError> {
        Ok(((), Vec::new()))
    }

    fn transform_stages(
        &self,
        _fitted: &Self::FittedStages,
        _block: &SKFeatureBlock<F>,
    ) -> Result<Vec<SKFeatureBlock<F>>, SKError> {
        Ok(Vec::new())
    }
}

/// Fit and probe one member stage, validating row alignment and recording width.
pub(super) fn sk_fit_one_member<F, Stage>(
    member_index: usize,
    stage: &Stage,
    block: &SKFeatureBlock<F>,
) -> Result<(Stage::Fitted, usize), SKError>
where
    F: SKFloat,
    Stage: SKPipelineTransformer<F>,
{
    let span = tracing::info_span!(
        "feature_union_member",
        member_index = member_index,
        operation = "fit"
    );
    let _guard = span.enter();
    let fitted = stage.fit_stage(block)?;
    let probe_output = stage.transform_stage(&fitted, block)?;
    let expected_rows = block.number_of_rows();
    let found_rows = probe_output.number_of_rows();
    let width = probe_output.number_of_columns();
    if found_rows != expected_rows {
        return Err(SKError::ShapeMismatch {
            expected: vec![expected_rows, width],
            found: vec![found_rows, width],
        });
    }
    Ok((fitted, width))
}

/// Transform `block` through one fitted member stage under its member span.
pub(super) fn sk_transform_one_member<F, Stage>(
    member_index: usize,
    stage: &Stage,
    fitted: &Stage::Fitted,
    block: &SKFeatureBlock<F>,
) -> Result<SKFeatureBlock<F>, SKError>
where
    F: SKFloat,
    Stage: SKPipelineTransformer<F>,
{
    let span = tracing::info_span!(
        "feature_union_member",
        member_index = member_index,
        operation = "transform"
    );
    let _guard = span.enter();
    stage.transform_stage(fitted, block)
}

macro_rules! impl_feature_union_tuple {
    ($count:expr; $( ($stage_type:ident, $stage_field:tt, $fitted_binding:ident, $width_binding:ident) ),+ ) => {
        impl<$( $stage_type ),+> SKFeatureUnionArity for ( $( $stage_type, )+ ) {
            fn number_of_members(&self) -> usize {
                $count
            }
        }

        impl<F, $( $stage_type ),+> SKFeatureUnionStages<F> for ( $( $stage_type, )+ )
        where
            F: SKFloat,
            $( $stage_type: SKPipelineTransformer<F>, )+
        {
            type FittedStages = ( $( $stage_type::Fitted, )+ );

            fn fit_stages(
                &self,
                block: &SKFeatureBlock<F>,
            ) -> Result<(Self::FittedStages, Vec<usize>), SKError> {
                $(
                    let ($fitted_binding, $width_binding) =
                        sk_fit_one_member($stage_field, &self.$stage_field, block)?;
                )+
                Ok((( $( $fitted_binding, )+ ), vec![ $( $width_binding ),+ ]))
            }

            fn transform_stages(
                &self,
                fitted: &Self::FittedStages,
                block: &SKFeatureBlock<F>,
            ) -> Result<Vec<SKFeatureBlock<F>>, SKError> {
                Ok(vec![
                    $( sk_transform_one_member($stage_field, &self.$stage_field, &fitted.$stage_field, block)? ),+
                ])
            }
        }
    };
}

impl_feature_union_tuple!(1; (Stage0, 0, fitted_0, width_0));
impl_feature_union_tuple!(2; (Stage0, 0, fitted_0, width_0), (Stage1, 1, fitted_1, width_1));
impl_feature_union_tuple!(3; (Stage0, 0, fitted_0, width_0), (Stage1, 1, fitted_1, width_1), (Stage2, 2, fitted_2, width_2));
impl_feature_union_tuple!(4; (Stage0, 0, fitted_0, width_0), (Stage1, 1, fitted_1, width_1), (Stage2, 2, fitted_2, width_2), (Stage3, 3, fitted_3, width_3));
impl_feature_union_tuple!(5; (Stage0, 0, fitted_0, width_0), (Stage1, 1, fitted_1, width_1), (Stage2, 2, fitted_2, width_2), (Stage3, 3, fitted_3, width_3), (Stage4, 4, fitted_4, width_4));
impl_feature_union_tuple!(6; (Stage0, 0, fitted_0, width_0), (Stage1, 1, fitted_1, width_1), (Stage2, 2, fitted_2, width_2), (Stage3, 3, fitted_3, width_3), (Stage4, 4, fitted_4, width_4), (Stage5, 5, fitted_5, width_5));
impl_feature_union_tuple!(7; (Stage0, 0, fitted_0, width_0), (Stage1, 1, fitted_1, width_1), (Stage2, 2, fitted_2, width_2), (Stage3, 3, fitted_3, width_3), (Stage4, 4, fitted_4, width_4), (Stage5, 5, fitted_5, width_5), (Stage6, 6, fitted_6, width_6));
impl_feature_union_tuple!(8; (Stage0, 0, fitted_0, width_0), (Stage1, 1, fitted_1, width_1), (Stage2, 2, fitted_2, width_2), (Stage3, 3, fitted_3, width_3), (Stage4, 4, fitted_4, width_4), (Stage5, 5, fitted_5, width_5), (Stage6, 6, fitted_6, width_6), (Stage7, 7, fitted_7, width_7));
