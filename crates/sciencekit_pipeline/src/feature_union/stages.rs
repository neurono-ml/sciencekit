//! Tuple-based member composition for [`super::SKFeatureUnion`] (spec
//! `feature-union`).

use sciencekit_common::{SKError, SKFloat};

use crate::{SKFeatureBlock, SKPipelineTransformer};

use super::errors::{SKFeatureUnionMemberError, sk_validate_probe_output};

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

    /// Fit every member on `block`, returning any member failure paired with
    /// its zero-based member index.
    fn fit_stages_checked(
        &self,
        block: &SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, Vec<usize>), SKFeatureUnionMemberError>;

    /// Transform `block` with every fitted member in declaration order,
    /// returning any member failure paired with its zero-based member index.
    fn transform_stages_checked(
        &self,
        fitted: &Self::FittedStages,
        block: &SKFeatureBlock<F>,
    ) -> Result<Vec<SKFeatureBlock<F>>, SKFeatureUnionMemberError>;

    /// Fit every member on `block`, converting any failure into [`SKError`].
    fn fit_stages(
        &self,
        block: &SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, Vec<usize>), SKError> {
        self.fit_stages_checked(block).map_err(Into::into)
    }

    /// Transform `block` with every fitted member in declaration order.
    fn transform_stages(
        &self,
        fitted: &Self::FittedStages,
        block: &SKFeatureBlock<F>,
    ) -> Result<Vec<SKFeatureBlock<F>>, SKError> {
        self.transform_stages_checked(fitted, block)
            .map_err(Into::into)
    }
}

impl SKFeatureUnionArity for () {
    fn number_of_members(&self) -> usize {
        0
    }
}

impl<F: SKFloat> SKFeatureUnionStages<F> for () {
    type FittedStages = ();

    fn fit_stages_checked(
        &self,
        _block: &SKFeatureBlock<F>,
    ) -> Result<(Self::FittedStages, Vec<usize>), SKFeatureUnionMemberError> {
        Ok(((), Vec::new()))
    }

    fn transform_stages_checked(
        &self,
        _fitted: &Self::FittedStages,
        _block: &SKFeatureBlock<F>,
    ) -> Result<Vec<SKFeatureBlock<F>>, SKFeatureUnionMemberError> {
        Ok(Vec::new())
    }
}

pub(super) fn sk_fit_one_member<F, Stage>(
    member_index: usize,
    stage: &Stage,
    block: &SKFeatureBlock<F>,
) -> Result<(Stage::Fitted, usize), SKFeatureUnionMemberError>
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
    let fitted = stage
        .fit_stage(block)
        .map_err(|error| SKFeatureUnionMemberError::member(member_index, error))?;
    let probe_output = stage
        .transform_stage(&fitted, block)
        .map_err(|error| SKFeatureUnionMemberError::member(member_index, error))?;
    let width = sk_validate_probe_output(member_index, &probe_output, block.number_of_rows())?;
    Ok((fitted, width))
}

pub(super) fn sk_transform_one_member<F, Stage>(
    member_index: usize,
    stage: &Stage,
    fitted: &Stage::Fitted,
    block: &SKFeatureBlock<F>,
) -> Result<SKFeatureBlock<F>, SKFeatureUnionMemberError>
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
    stage
        .transform_stage(fitted, block)
        .map_err(|error| SKFeatureUnionMemberError::member(member_index, error))
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

            fn fit_stages_checked(
                &self,
                block: &SKFeatureBlock<F>,
            ) -> Result<(Self::FittedStages, Vec<usize>), SKFeatureUnionMemberError> {
                $(
                    let ($fitted_binding, $width_binding) =
                        sk_fit_one_member($stage_field, &self.$stage_field, block)?;
                )+
                Ok((( $( $fitted_binding, )+ ), vec![ $( $width_binding ),+ ]))
            }

            fn transform_stages_checked(
                &self,
                fitted: &Self::FittedStages,
                block: &SKFeatureBlock<F>,
            ) -> Result<Vec<SKFeatureBlock<F>>, SKFeatureUnionMemberError> {
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
