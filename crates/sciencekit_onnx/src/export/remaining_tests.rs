//! Neighbour, clustering and decomposition parity tests.

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKClustererPredictor,
    SKRegressorPredictor, SKSupervisedFit, SKUnsupervisedFit,
};

use crate::acceptance_tests::write_fixture;

use super::SKToOnnx;

fn features() -> Array2<f64> {
    array![
        [1.0, 8.0, 2.0],
        [1.0, 0.0, 5.0],
        [9.0, 1.0, 1.0],
        [9.0, 6.0, 2.0],
        [1.0, 7.0, 2.0],
        [9.0, 0.0, 5.0]
    ]
}

fn json_matrix(matrix: &Array2<f64>) -> serde_json::Value {
    serde_json::json!(
        matrix
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f64>>>()
    )
}

#[test]
fn export_remaining_neighbors_classifier_matches_the_rust_probabilities() {
    use sciencekit_neighbors::SKKNeighborsClassifier;

    let features = features();
    let labels = ["left", "left", "right", "right", "left", "right"];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the classifier fits");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("the classifier predicts");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    write_fixture(
        "knn_classifier_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({
            "label": predicted.to_vec(),
            "probabilities": json_matrix(&probabilities),
        }),
        1e-9,
    );
}

#[test]
fn export_remaining_neighbors_regressor_matches_the_rust_prediction() {
    use sciencekit_neighbors::SKKNeighborsRegressor;

    let features = features();
    let targets = array![1.0_f64, 2.0, 9.0, 8.0, 1.5, 8.5];
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .expect("the regressor configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the regressor fits");
    let prediction = model
        .predict(features.view())
        .expect("the regressor predicts");
    let onnx = model.to_onnx().expect("the regressor exports");
    write_fixture(
        "knn_regressor_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): prediction.iter().map(|value| vec![*value]).collect::<Vec<Vec<f64>>>() }),
        1e-9,
    );
}

#[test]
fn export_remaining_kmeans_matches_the_rust_labels() {
    use sciencekit_cluster::kmeans::SKKMeans;

    let features = features();
    let model = SKKMeans::build()
        .number_of_clusters(2)
        .random_seed(3)
        .build()
        .expect("the clustering configuration is valid")
        .fit(&features)
        .expect("the clusters fit");
    let predicted = model
        .predict_clusters(features.view())
        .expect("the clusters predict");
    let onnx = model.to_onnx().expect("the clusters export");
    write_fixture(
        "kmeans_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): predicted.iter().map(|value| value.to_i64()).collect::<Vec<i64>>() }),
        0.0,
    );
}

#[test]
fn export_remaining_pca_matches_the_rust_transform() {
    use sciencekit_decomposition::pca::SKPCA;

    let features = features();
    let model = SKPCA::<f64>::build()
        .number_of_components(2)
        .build()
        .expect("the decomposition configuration is valid")
        .fit(&features)
        .expect("the decomposition fits");
    let transformed = model.transform(&features).expect("the transform");
    let onnx = model.to_onnx().expect("the decomposition exports");
    write_fixture(
        "pca_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): json_matrix(&transformed) }),
        1e-9,
    );
}

#[test]
fn export_remaining_truncated_svd_matches_the_rust_transform() {
    use sciencekit_decomposition::truncated_svd::SKTruncatedSVD;

    let features = features();
    let model = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .build()
        .expect("the decomposition configuration is valid")
        .fit(&features)
        .expect("the decomposition fits");
    let transformed = model.transform(&features).expect("the transform");
    let onnx = model.to_onnx().expect("the decomposition exports");
    write_fixture(
        "truncated_svd_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): json_matrix(&transformed) }),
        1e-9,
    );
}
