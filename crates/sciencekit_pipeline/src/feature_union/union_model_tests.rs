//! Accessor and thread-safety tests for `SKFeatureUnionModel` (spec
//! `feature-union`, Task 2.3).

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKBuilder, SKDataView, SKExecutionMode, SKFeatureBlock};
use sciencekit_impute::missing_indicator::SKMissingIndicator;
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::feature_union::{SKBlockTransformer, SKFeatureUnion, SKFeatureUnionModel};

fn dense_values(block: &SKFeatureBlock<f64>) -> Array2<f64> {
    match block.view() {
        SKDataView::Dense(view) => view.to_owned(),
        _ => panic!("expected dense block"),
    }
}

#[test]
fn accessors_report_member_count_widths_and_summed_total_width() {
    let input = SKFeatureBlock::from(array![[1.0_f64, f64::NAN], [3.0, 4.0], [5.0, 6.0]]);
    let union = SKFeatureUnion::build()
        .stages((
            SKSimpleImputer::<f64>::build().build().unwrap(),
            SKMissingIndicator::<f64>::build()
                .error_on_new(false)
                .build()
                .unwrap(),
            SKSimpleImputer::<f64>::build().build().unwrap(),
        ))
        .backend(SKBackendKind::Faer)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .unwrap();

    let fitted = union.fit(&input).unwrap();
    assert_eq!(fitted.number_of_members(), 3);
    assert_eq!(fitted.member_widths(), &[2, 1, 2]);
    assert_eq!(fitted.number_of_features(), 5);
    assert_eq!(fitted.number_of_output_features(), 5);
    assert_eq!(fitted.number_of_input_features(), 2);
    assert_eq!(
        fitted.member_widths().iter().copied().sum::<usize>(),
        fitted.number_of_features()
    );
    assert_eq!(fitted.backend(), SKBackendKind::Faer);
    assert_eq!(fitted.execution_mode(), SKExecutionMode::Automatic);
    assert_eq!(fitted.stages().0.backend(), SKBackendKind::Faer);
}

#[test]
fn non_default_backend_and_execution_mode_propagate_to_fitted_model() {
    let input = SKFeatureBlock::from(array![[1.0_f64, 2.0], [3.0, 4.0]]);
    let union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert_eq!(union.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        union.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let fitted = union.fit(&input).unwrap();
    assert_eq!(fitted.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        fitted.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn fitted_tuple_and_boxed_unions_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKFeatureUnionModel<f64, (SKStandardScaler, SKMinMaxScaler<f64>)>>();
    assert_send_sync::<SKFeatureUnionModel<f64, Vec<Box<dyn SKBlockTransformer<f64>>>>>();
}

#[test]
fn concurrent_transform_across_threads_agrees_with_sequential_result() {
    let training = SKFeatureBlock::from(array![
        [1.0_f64, 10.0],
        [3.0, 20.0],
        [5.0, 30.0],
        [7.0, 40.0]
    ]);
    let query = SKFeatureBlock::from(array![[2.0_f64, 15.0], [6.0, 35.0]]);

    let fitted = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap()
        .fit(&training)
        .unwrap();

    let expected = dense_values(&fitted.transform(&query).unwrap());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let model = &fitted;
                let query_ref = &query;
                scope.spawn(move || dense_values(&model.transform(query_ref).unwrap()))
            })
            .collect();
        handles.into_iter().for_each(|handle| {
            assert_eq!(handle.join().unwrap(), expected);
        });
    });
}
