//! What one exporter stage produced, so composite models can wire stages.

/// A tensor a stage appended to the graph.
#[derive(Debug, Clone)]
pub(crate) struct SKOnnxStageOutput {
    pub(crate) tensor: String,
    pub(crate) width: usize,
    pub(crate) kind: SKOnnxStageKind,
}

/// How the produced tensor may be consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SKOnnxStageKind {
    /// A feature matrix a further stage can consume.
    Transformed,
    /// A single-column regression prediction.
    Regression,
    /// A label output plus a probability matrix.
    Classification {
        label_tensor: String,
        probability_tensor: String,
        class_count: usize,
    },
    /// A label output plus raw decision values (score-based classifiers).
    DecisionValues {
        label_tensor: String,
        decision_tensor: String,
        decision_width: usize,
    },
    /// A single integer label output (clustering).
    Labels {
        label_tensor: String,
        label_count: usize,
    },
}
