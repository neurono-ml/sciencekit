//! Flat-tuple extension trait for [`super::SKFeatureUnionBuilder::with_stage`]
//! (spec `feature-union`).

/// Append one stage to a flat tuple of up to eight union members.
pub trait SKAppendUnionStage<Next> {
    /// The extended tuple type containing `Next` at the end.
    type Extended;

    /// Consume `self` and append `next`.
    fn append_union_stage(self, next: Next) -> Self::Extended;
}

impl<Next> SKAppendUnionStage<Next> for () {
    type Extended = (Next,);

    fn append_union_stage(self, next: Next) -> Self::Extended {
        (next,)
    }
}

macro_rules! impl_append_union_stage {
    ( $( ($stage_type:ident, $stage_field:tt) ),+ ) => {
        impl<$( $stage_type, )+ Next> SKAppendUnionStage<Next> for ( $( $stage_type, )+ ) {
            type Extended = ( $( $stage_type, )+ Next );

            fn append_union_stage(self, next: Next) -> Self::Extended {
                ( $( self.$stage_field, )+ next )
            }
        }
    };
}

impl_append_union_stage!((Stage0, 0));
impl_append_union_stage!((Stage0, 0), (Stage1, 1));
impl_append_union_stage!((Stage0, 0), (Stage1, 1), (Stage2, 2));
impl_append_union_stage!((Stage0, 0), (Stage1, 1), (Stage2, 2), (Stage3, 3));
impl_append_union_stage!(
    (Stage0, 0),
    (Stage1, 1),
    (Stage2, 2),
    (Stage3, 3),
    (Stage4, 4)
);
impl_append_union_stage!(
    (Stage0, 0),
    (Stage1, 1),
    (Stage2, 2),
    (Stage3, 3),
    (Stage4, 4),
    (Stage5, 5)
);
impl_append_union_stage!(
    (Stage0, 0),
    (Stage1, 1),
    (Stage2, 2),
    (Stage3, 3),
    (Stage4, 4),
    (Stage5, 5),
    (Stage6, 6)
);
