//! Dynamic boxed fallback tests for `SKFeatureUnion` (spec `feature-union`,
//! Task 2.1).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKDataView, SKFeatureBlock};
use sciencekit_impute::missing_indicator::SKMissingIndicator;
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::feature_union::{SKBlockTransformer, SKFeatureUnion};

#[test]
fn boxed_dynamic_union_matches_tuple_union_on_dense_outputs() {
    let training = SKFeatureBlock::from(array![[1.0_f64, 10.0], [3.0, 20.0], [5.0, 30.0]]);
    let query = SKFeatureBlock::from(array![[2.0_f64, 15.0], [4.0, 25.0]]);

    let tuple_union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();

    let boxed_list: Vec<Box<dyn SKBlockTransformer<f64>>> = vec![
        Box::new(SKStandardScaler::build().build().unwrap()),
        Box::new(SKMinMaxScaler::<f64>::build().build().unwrap()),
    ];
    let boxed_union = SKFeatureUnion::build().stages(boxed_list).build().unwrap();

    let tuple_output = tuple_union
        .fit(&training)
        .unwrap()
        .transform(&query)
        .unwrap();
    let boxed_output = boxed_union
        .fit(&training)
        .unwrap()
        .transform(&query)
        .unwrap();

    match (tuple_output.view(), boxed_output.view()) {
        (SKDataView::Dense(expected), SKDataView::Dense(actual)) => assert_eq!(expected, actual),
        _ => panic!("both unions must emit dense matrices"),
    }
}

#[test]
fn boxed_builder_matches_tuple_union_on_mixed_csr_outputs() {
    let input = SKFeatureBlock::from(array![[1.0_f64, f64::NAN], [3.0, 4.0], [5.0, 6.0]]);

    let tuple_union = SKFeatureUnion::build()
        .stages((
            SKSimpleImputer::<f64>::build().build().unwrap(),
            SKMissingIndicator::<f64>::build()
                .sparse_output(true)
                .error_on_new(false)
                .build()
                .unwrap(),
        ))
        .build()
        .unwrap();

    let boxed_union = SKFeatureUnion::boxed::<f64>()
        .with_boxed_stage(SKSimpleImputer::<f64>::build().build().unwrap())
        .with_boxed_stage(
            SKMissingIndicator::<f64>::build()
                .sparse_output(true)
                .error_on_new(false)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let tuple_output = tuple_union.fit(&input).unwrap().transform(&input).unwrap();
    let boxed_output = boxed_union.fit(&input).unwrap().transform(&input).unwrap();

    match (tuple_output.view(), boxed_output.view()) {
        (SKDataView::Sparse(expected), SKDataView::Sparse(actual)) => {
            assert_eq!(expected.to_dense(), actual.to_dense());
        }
        _ => panic!("both unions must emit CSR sparse matrices"),
    }
}

#[test]
fn boxed_dynamic_union_supports_more_than_eight_members() {
    let input = SKFeatureBlock::from(array![[1.0_f64], [3.0]]);
    let members: Vec<Box<dyn SKBlockTransformer<f64>>> = (0..10)
        .map(|_| -> Box<dyn SKBlockTransformer<f64>> {
            Box::new(SKStandardScaler::build().build().unwrap())
        })
        .collect();

    let union = SKFeatureUnion::build().stages(members).build().unwrap();
    assert_eq!(union.number_of_members(), 10);
    let output = union.fit(&input).unwrap().transform(&input).unwrap();
    assert_eq!(
        (output.number_of_rows(), output.number_of_columns()),
        (2, 10)
    );
}
