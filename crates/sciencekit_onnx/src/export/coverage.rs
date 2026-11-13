//! The documented family/operator coverage matrix (spec `onnx-export`,
//! requirement "Per-family operators").
//!
//! The matrix is the single source of truth for what the exporter supports. It
//! is exposed as data so tests can assert that a family advertised here actually
//! exports, and so the crate documentation can be kept truthful.

/// One row of the coverage matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKOnnxFamilyCoverage {
    /// The human-readable family name.
    pub family: &'static str,
    /// The estimator names this family covers.
    pub estimators: &'static [&'static str],
    /// The ONNX operators the family emits.
    pub operators: &'static [&'static str],
    /// Whether the exporter covers the family today.
    pub supported: bool,
    /// The documented limitation or the alternative path, when any.
    pub note: &'static str,
}

/// The family/operator coverage matrix.
pub fn sk_onnx_coverage() -> &'static [SKOnnxFamilyCoverage] {
    &[
        SKOnnxFamilyCoverage {
            family: "preprocessing.scalers",
            estimators: &["SKStandardScaler", "SKMinMaxScaler", "SKRobustScaler"],
            operators: &["Sub", "Mul", "Div", "Add", "Clip"],
            supported: true,
            note: "per-column statistics broadcast over the batch dimension",
        },
        SKOnnxFamilyCoverage {
            family: "preprocessing.encoders",
            estimators: &["SKOneHotEncoder"],
            operators: &["Gather", "Equal", "Cast", "Concat"],
            supported: true,
            note: "numeric integer categories and the ignore policy only; nominal string categories and the reject policy return UnsupportedOperation",
        },
        SKOnnxFamilyCoverage {
            family: "neighbors",
            estimators: &["SKKNeighborsClassifier", "SKKNeighborsRegressor"],
            operators: &[
                "Unsqueeze",
                "Sub",
                "Mul",
                "ReduceSum",
                "Neg",
                "TopK",
                "Gather",
                "Equal",
                "Cast",
                "Concat",
                "Div",
                "ArgMax",
                "ReduceMean",
            ],
            supported: true,
            note: "dense brute-force search with uniform weights and the euclidean metric",
        },
        SKOnnxFamilyCoverage {
            family: "clustering",
            estimators: &["SKKMeans", "SKMiniBatchKMeans"],
            operators: &["Unsqueeze", "Sub", "Mul", "ReduceSum", "ArgMin"],
            supported: true,
            note: "prediction is the nearest centre; the label output is the cluster index",
        },
        SKOnnxFamilyCoverage {
            family: "decomposition",
            estimators: &["SKPCA", "SKTruncatedSVD"],
            operators: &["Sub", "MatMul", "Div"],
            supported: true,
            note: "PCA includes the centring mean and the whitening scale when configured",
        },
        SKOnnxFamilyCoverage {
            family: "outlier",
            estimators: &["SKIsolationForest", "SKDBSCAN"],
            operators: &[],
            supported: false,
            note: "the isolation-forest score composes per-tree path lengths and DBSCAN has no fitted prediction; both stay out of the matrix until their formulas have parity fixtures",
        },
        SKOnnxFamilyCoverage {
            family: "svm",
            estimators: &["SKSVC", "SKSVR", "SKLinearSVC"],
            operators: &[
                "MatMul",
                "Add",
                "Unsqueeze",
                "Sub",
                "Mul",
                "ReduceSum",
                "Neg",
                "Exp",
                "Tanh",
                "Pow",
                "Greater",
                "Cast",
                "Concat",
                "Gather",
                "Abs",
                "ArgMax",
            ],
            supported: true,
            note: "linear, polynomial, rbf and sigmoid kernels with dense support vectors; precomputed kernels and CSR support-vector blocks return UnsupportedOperation naming the reason, and SVC exports labels plus one-vs-one decision values (not calibrated probabilities)",
        },
        SKOnnxFamilyCoverage {
            family: "composites",
            estimators: &["SKPipelineClassifier", "SKPipelineRegressor"],
            operators: &["composed stage subgraphs"],
            supported: true,
            note: "stages are recovered from their documented tensor signatures and wired in order; the final estimator produces the outputs",
        },
        SKOnnxFamilyCoverage {
            family: "trees",
            estimators: &[
                "SKDecisionTreeClassifier",
                "SKDecisionTreeRegressor",
                "SKRandomForestClassifier",
                "SKRandomForestRegressor",
            ],
            operators: &["TreeEnsembleClassifier", "TreeEnsembleRegressor", "ArgMax"],
            supported: true,
            note: "f32 only: the ONNX-ML schema stores thresholds and leaf weights as float attributes, so double trees return UnsupportedOperation",
        },
        SKOnnxFamilyCoverage {
            family: "boosting",
            estimators: &[
                "SKGradientBoostingClassifier",
                "SKGradientBoostingRegressor",
                "SKHistGradientBoostingClassifier",
                "SKHistGradientBoostingRegressor",
            ],
            operators: &[],
            supported: false,
            note: "boosting states carry a base score per class and a learning rate; the exporter keeps them out of the matrix until that composition has parity fixtures",
        },
        SKOnnxFamilyCoverage {
            family: "linear.regressors",
            estimators: &[
                "SKLinearRegression",
                "SKRidge",
                "SKLasso",
                "SKElasticNet",
                "SKSGDRegressor",
            ],
            operators: &["MatMul", "Add"],
            supported: true,
            note: "one output column; the coordinate-descent families require a single configured alpha",
        },
        SKOnnxFamilyCoverage {
            family: "linear.classifiers",
            estimators: &["SKLogisticRegression"],
            operators: &[
                "MatMul",
                "Add",
                "Sigmoid",
                "Softmax",
                "ReduceSum",
                "Div",
                "Concat",
                "ArgMax",
            ],
            supported: true,
            note: "binary, multinomial and one-versus-rest strategies; the label output is the canonical class index",
        },
    ]
}

/// The estimator names with a working ONNX exporter.
pub fn sk_onnx_supported_families() -> Vec<&'static str> {
    sk_onnx_coverage()
        .iter()
        .filter(|row| row.supported)
        .flat_map(|row| row.estimators.iter().copied())
        .collect()
}
