//! Format-agnostic export of a fitted feature union.
//!
//! A fitted union owns a static tuple of member models, so their states nest
//! under `feature_union.member.<index>`. The dynamic `Vec<Box<dyn ...>>` form
//! stores type-erased members and is deliberately not exportable.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExportable, SKFloat, SKImportable, SKModelMetadata,
    SKModelState, SKParameterValue, sk_backend_parameter, sk_embed_state,
    sk_model_tensor_data_type,
};

use crate::feature_union::model::SKFeatureUnionModel;
use crate::feature_union::stages::SKFeatureUnionStages;

/// Stable estimator identifier for exported feature-union states.
pub const SK_FEATURE_UNION_ESTIMATOR_NAME: &str = "SKFeatureUnion";

impl<F, Stages> SKExportable<F> for SKFeatureUnionModel<F, Stages>
where
    F: SKFloat,
    Stages: SKFeatureUnionStages<F>,
    Stages::FittedStages: SKFeatureUnionExportableStages<F>,
{
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let stage_states = self.fitted_stages.export_member_states()?;
        let mut metadata = SKModelMetadata::new(
            SK_FEATURE_UNION_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.number_of_input_features,
        )?
        .with_parameter(
            "number_of_members",
            SKParameterValue::from(stage_states.len() as i64),
        )?
        .with_parameter(
            "number_of_features",
            SKParameterValue::from(self.number_of_features as i64),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let widths: Vec<i64> = self
            .member_widths
            .iter()
            .map(|&width| width as i64)
            .collect();
        let mut tensors = vec![sciencekit_common::SKTensor::integer(
            "feature_union.member_widths",
            vec![widths.len()],
            widths,
        )?];
        for (index, state) in stage_states.iter().enumerate() {
            metadata = sk_embed_state(
                metadata,
                &mut tensors,
                &format!("feature_union.member.{index}"),
                state,
            )?;
        }
        SKModelState::new(metadata, tensors)
    }
}

/// Read the member count of an exported feature-union state.
///
/// # Errors
///
/// Returns the first [`SKError`] raised by validation.
pub fn sk_feature_union_member_count<F: SKFloat>(state: &SKModelState<F>) -> Result<i64, SKError> {
    sciencekit_common::sk_validate_import_metadata(
        state,
        SK_FEATURE_UNION_ESTIMATOR_NAME,
        SK_MODEL_FORMAT_VERSION,
        state.metadata().feature_count(),
    )?;
    Ok(sciencekit_common::sk_integer_parameter(
        state.metadata(),
        "number_of_members",
        0,
    ))
}

/// Export every fitted member of a flat feature-union tuple in order.
///
/// The feature union stores its fitted members as a flat tuple (unlike the
/// typed pipeline's nested tuple), so it owns this dedicated trait.
pub trait SKFeatureUnionExportableStages<F: SKFloat> {
    /// Export every fitted member state in declaration order.
    ///
    /// # Errors
    ///
    /// Returns the first member export failure.
    fn export_member_states(&self) -> Result<Vec<SKModelState<F>>, SKError>;
}

macro_rules! impl_feature_union_exportable_tuple {
    ($( ($member_type:ident, $member_binding:ident) ),+) => {
        impl<F, $( $member_type ),+> SKFeatureUnionExportableStages<F> for ( $( $member_type, )+ )
        where
            F: SKFloat,
            $( $member_type: SKExportable<F>, )+
        {
            fn export_member_states(&self) -> Result<Vec<SKModelState<F>>, SKError> {
                let ( $( $member_binding, )+ ) = self;
                Ok(vec![ $( $member_binding.export_state()?, )+ ])
            }
        }
    };
}

impl_feature_union_exportable_tuple!((Member0, member0));
impl_feature_union_exportable_tuple!((Member0, member0), (Member1, member1));
impl_feature_union_exportable_tuple!((Member0, member0), (Member1, member1), (Member2, member2));
impl_feature_union_exportable_tuple!(
    (Member0, member0),
    (Member1, member1),
    (Member2, member2),
    (Member3, member3)
);
impl_feature_union_exportable_tuple!(
    (Member0, member0),
    (Member1, member1),
    (Member2, member2),
    (Member3, member3),
    (Member4, member4)
);
impl_feature_union_exportable_tuple!(
    (Member0, member0),
    (Member1, member1),
    (Member2, member2),
    (Member3, member3),
    (Member4, member4),
    (Member5, member5)
);
impl_feature_union_exportable_tuple!(
    (Member0, member0),
    (Member1, member1),
    (Member2, member2),
    (Member3, member3),
    (Member4, member4),
    (Member5, member5),
    (Member6, member6)
);
impl_feature_union_exportable_tuple!(
    (Member0, member0),
    (Member1, member1),
    (Member2, member2),
    (Member3, member3),
    (Member4, member4),
    (Member5, member5),
    (Member6, member6),
    (Member7, member7)
);

impl<F, Stages> SKImportable<F> for SKFeatureUnionModel<F, Stages>
where
    F: SKFloat,
    Stages: SKFeatureUnionStages<F>,
{
    fn import_state(_state: &SKModelState<F>) -> Result<Self, SKError> {
        Err(SKError::UnsupportedOperation {
            operation: "import_state",
            reason: "a feature-union state nests every fitted member, but the configured member \
                     chain holds live estimator values rather than data; rebuild the union in \
                     code and import each nested member state"
                .to_string(),
        })
    }
}
