//! SVM parity tests: kernel expansions against the Rust decisions/labels and
//! the reference ONNX runtime.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_svm::{SKKernel, SKKernelGamma};

use crate::acceptance_tests::write_fixture;

use super::SKToOnnx;

fn binary_features<F: sciencekit_common::SKFloat>() -> Array2<F> {
    array![
        [-2.0, -1.0],
        [-1.5, -0.5],
        [-1.0, -1.5],
        [1.0, 1.0],
        [1.5, 0.5],
        [2.0, 1.5]
    ]
    .mapv(|value| F::from(value).expect("the fixture is finite"))
}

fn multiclass_features<F: sciencekit_common::SKFloat>() -> Array2<F> {
    array![
        [-2.0, -1.0],
        [-1.5, -0.5],
        [2.0, 2.0],
        [2.5, 1.5],
        [-2.0, 2.5],
        [-1.5, 3.0]
    ]
    .mapv(|value| F::from(value).expect("the fixture is finite"))
}

fn json_matrix<F: sciencekit_common::SKFloat>(matrix: &Array2<F>) -> serde_json::Value {
    serde_json::json!(
        matrix
            .rows()
            .into_iter()
            .map(|row| row
                .iter()
                .map(|value| value.to_f64().expect("the fixture is finite"))
                .collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>()
    )
}

fn write_classifier_fixture<F: sciencekit_common::SKFloat>(
    name: &str,
    model: &crate::graph::SKOnnxModel,
    features: &Array2<F>,
    labels: &Array1<i64>,
    decisions: &Array2<F>,
    tolerance: f64,
) {
    write_fixture(
        name,
        model,
        "features",
        json_matrix(features),
        serde_json::json!({
            model.outputs()[0].name(): labels.to_vec(),
            model.outputs()[1].name(): json_matrix(decisions),
        }),
        tolerance,
    );
}

#[test]
fn export_svm_binary_svc_rbf_matches_the_rust_decisions_and_labels() {
    use sciencekit_svm::svc::SKSVC;

    let features: Array2<f64> = binary_features();
    let labels = ["left", "left", "left", "right", "right", "right"];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(0.5_f64))
        .probability(false)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let decisions = model
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    write_classifier_fixture(
        "svc_binary_rbf_f64",
        &onnx,
        &features,
        &predicted,
        &decisions,
        1e-8,
    );
}

#[test]
fn export_svm_binary_svc_linear_matches_the_rust_decisions_and_labels() {
    use sciencekit_svm::svc::SKSVC;

    let features: Array2<f64> = binary_features();
    let labels = ["left", "left", "left", "right", "right", "right"];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .probability(false)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let decisions = model
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    write_classifier_fixture(
        "svc_binary_linear_f64",
        &onnx,
        &features,
        &predicted,
        &decisions,
        1e-9,
    );
}

#[test]
fn export_svm_multiclass_svc_one_vs_one_matches_the_rust_outputs() {
    use sciencekit_svm::svc::SKSVC;

    let features: Array2<f64> = multiclass_features();
    let labels = ["a", "a", "b", "b", "c", "c"];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(0.5_f64))
        .probability(false)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let decisions = model
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    write_classifier_fixture(
        "svc_multiclass_rbf_f64",
        &onnx,
        &features,
        &predicted,
        &decisions,
        1e-8,
    );
}

#[test]
fn export_svm_svr_rbf_matches_the_rust_prediction() {
    use sciencekit_svm::svr::SKSVR;

    let features: Array2<f64> = binary_features();
    let targets: Array1<f64> = array![-3.0, -2.4, -2.0, 2.5, 3.0, 4.0];
    let model = SKSVR::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(0.5_f64))
        .epsilon(0.1_f64)
        .build()
        .expect("the regressor configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the regressor fits");
    let prediction = model
        .predict(features.view())
        .expect("the regressor predicts");
    let onnx = model.to_onnx().expect("the regressor exports");
    write_fixture(
        "svr_rbf_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): prediction.iter().map(|value| vec![*value]).collect::<Vec<Vec<f64>>>() }),
        1e-8,
    );
}

#[test]
fn export_svm_linear_svc_binary_and_multiclass_match_the_rust_outputs() {
    use sciencekit_svm::linear_svc::SKLinearSVC;

    let features: Array2<f64> = multiclass_features();
    let binary_labels = ["left", "left", "right", "right", "left", "left"];
    let binary = SKLinearSVC::<f64>::build()
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), binary_labels.as_slice())
        .expect("the classifier fits");
    let decisions = binary
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = binary.predict_labels(features.view()).expect("the labels");
    let onnx = binary.to_onnx().expect("the classifier exports");
    write_classifier_fixture(
        "linear_svc_binary_f64",
        &onnx,
        &features,
        &predicted,
        &decisions,
        1e-9,
    );

    let labels = ["a", "a", "b", "b", "c", "c"];
    let multiclass = SKLinearSVC::<f64>::build()
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let decisions = multiclass
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = multiclass
        .predict_labels(features.view())
        .expect("the labels");
    let onnx = multiclass.to_onnx().expect("the classifier exports");
    write_classifier_fixture(
        "linear_svc_multiclass_f64",
        &onnx,
        &features,
        &predicted,
        &decisions,
        1e-9,
    );
}

#[test]
fn export_svm_precomputed_kernel_is_rejected_by_name() {
    use sciencekit_common::SKError;
    use sciencekit_svm::svc::SKSVC;

    // A precomputed gram matrix lets SVC fit without a feature kernel; the
    // exported graph cannot read a gram query, so it must refuse explicitly.
    let gram: Array2<f64> = Array2::eye(4);
    let labels = ["a", "a", "b", "b"];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::Precomputed)
        .build()
        .expect("the classifier configuration is valid")
        .fit(gram.view(), labels.as_slice())
        .expect("the classifier fits");
    let error = model
        .to_onnx()
        .expect_err("precomputed kernels are rejected");
    match error {
        SKError::UnsupportedOperation { reason, .. } => {
            assert!(reason.contains("precomputed"), "reason: {reason}");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn export_svm_binary_svc_non_default_gamma_matches_the_rust_outputs() {
    use sciencekit_svm::svc::SKSVC;

    let features: Array2<f64> = binary_features();
    let labels = ["left", "left", "left", "right", "right", "right"];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(1.75_f64))
        .probability(false)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let decisions = model
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    write_classifier_fixture(
        "svc_binary_rbf_gamma_f64",
        &onnx,
        &features,
        &predicted,
        &decisions,
        1e-8,
    );
}

#[test]
fn export_svm_multiclass_svc_vote_tie_matches_the_rust_outputs() {
    use sciencekit_svm::svc::SKSVC;

    let training: Array2<f64> = array![
        [-2.0, 0.0],
        [-1.5, 0.5],
        [2.0, 0.0],
        [1.5, -0.5],
        [0.0, 2.0],
        [0.5, 1.5]
    ];
    let labels = ["a", "a", "b", "b", "c", "c"];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(0.5_f64))
        .probability(false)
        .build()
        .expect("the classifier configuration is valid")
        .fit(training.view(), labels.as_slice())
        .expect("the classifier fits");
    // Far from every support vector the three pairwise margins are all
    // negative, so each class receives exactly one vote: a 1-1-1 tie that the
    // Rust winner rule resolves to the smallest class index.
    let mut features = training.clone();
    features
        .push_row(array![-3.0, -3.0].view())
        .expect("the row appends");
    features
        .push_row(array![3.0, 3.0].view())
        .expect("the row appends");
    let decisions = model
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    write_classifier_fixture(
        "svc_multiclass_vote_tie_f64",
        &onnx,
        &features,
        &predicted,
        &decisions,
        1e-8,
    );
}

#[test]
fn export_svm_linear_svc_explicit_class_weights_match_the_rust_outputs() {
    use sciencekit_svm::linear_svc::SKLinearSVC;
    use sciencekit_svm::svc::SKClassWeight;

    let features: Array2<f64> = multiclass_features();
    let labels = ["left", "left", "right", "right", "left", "left"];
    let model = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0_f64, 4.0]))
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let decisions = model
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    write_classifier_fixture(
        "linear_svc_binary_weighted_f64",
        &onnx,
        &features,
        &predicted,
        &decisions,
        1e-9,
    );
}

#[test]
fn export_svm_f32_classifier_declares_float_tensors_and_matches_the_rust_outputs() {
    use sciencekit_svm::svc::SKSVC;

    let features: Array2<f32> = binary_features();
    let labels = ["left", "left", "left", "right", "right", "right"];
    let model = SKSVC::<f32>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(0.5_f32))
        .probability(false)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let decisions = model
        .decision_function(features.view())
        .expect("the decisions");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    assert_eq!(
        onnx.inputs()[0].element_type(),
        crate::graph::SKOnnxTensorDType::Float
    );
    let input = serde_json::json!(
        features
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f32>>>()
    );
    let decisions_json = serde_json::json!(
        decisions
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f32>>>()
    );
    write_fixture(
        "svc_binary_rbf_f32",
        &onnx,
        "features",
        input,
        serde_json::json!({
            onnx.outputs()[0].name(): predicted.to_vec(),
            onnx.outputs()[1].name(): decisions_json,
        }),
        1e-4,
    );
}
