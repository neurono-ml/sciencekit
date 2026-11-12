//! End-to-end acceptance tests across dense and sparse CSR representations
//! and OTLP pipeline builders (spec `telemetry-export`).

use ndarray::array;
use opentelemetry_sdk::logs::InMemoryLogExporter;
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::trace::InMemorySpanExporter;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKDataView, SKError, SKExecutionMode, SKOperationAttributes,
    SKOperationKind, sk_run_operation,
};
use sprs::TriMat;

use crate::{SKTelemetryConfiguration, SKTelemetryPipeline};

#[test]
fn dense_and_sparse_csr_operations_export_matching_telemetry_signals() {
    let span_exporter = InMemorySpanExporter::default();
    let log_exporter = InMemoryLogExporter::default();
    let metric_exporter = InMemoryMetricExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-parity-test")
        .service_version("0.7.0")
        .build()
        .expect("configuration builds");
    let pipeline = SKTelemetryPipeline::from_exporters(
        configuration,
        span_exporter.clone(),
        log_exporter.clone(),
        metric_exporter.clone(),
    );

    let dense_matrix = array![[1.0_f64, 0.0, 2.0], [0.0, 3.0, 0.0]];
    let mut triplets = TriMat::<f64>::new((2, 3));
    triplets.add_triplet(0, 0, 1.0);
    triplets.add_triplet(0, 2, 2.0);
    triplets.add_triplet(1, 1, 3.0);
    let csr_matrix = triplets.to_csr::<usize>();

    tracing::subscriber::with_default(pipeline.subscriber(), || {
        let dense_view: SKDataView<'_, f64> = (&dense_matrix).try_into().expect("dense view");
        let sparse_view: SKDataView<'_, f64> = csr_matrix.view().try_into().expect("csr view");

        for view in [dense_view, sparse_view] {
            let representation = view.representation();
            let (rows_count, columns_count) = match view {
                SKDataView::Dense(dense) => dense.dim(),
                SKDataView::Sparse(sparse) => sparse.shape(),
                _ => unreachable!("known feature views"),
            };
            let outcome = sk_run_operation(
                SKOperationAttributes {
                    operation: SKOperationKind::Transform,
                    rows_count,
                    columns_count,
                    execution_mode: SKExecutionMode::InProcessSynchronous,
                    backend: SKBackendKind::Faer,
                },
                || -> Result<usize, SKError> {
                    tracing::info!(
                        representation = representation,
                        "processed view without densifying"
                    );
                    Ok(rows_count * columns_count)
                },
            );
            assert_eq!(outcome.expect("operation succeeds"), 6);
        }
    });

    pipeline.force_flush().expect("flush succeeds");
    assert_eq!(span_exporter.get_finished_spans().unwrap().len(), 2);
    assert_eq!(log_exporter.get_emitted_logs().unwrap().len(), 2);
    assert!(!metric_exporter.get_finished_metrics().unwrap().is_empty());
    pipeline.shutdown().expect("shutdown succeeds");
}

#[cfg(feature = "telemetry")]
#[test]
fn otlp_grpc_and_http_pipelines_construct_and_shut_down_cleanly() {
    use crate::{SKTelemetryTransport, sk_build_otlp_pipeline};

    for transport in [SKTelemetryTransport::Grpc, SKTelemetryTransport::Http] {
        let configuration = SKTelemetryConfiguration::builder()
            .service_name("sciencekit-otlp-smoke")
            .transport(transport)
            .header("x-api-key", "smoke-token")
            .build()
            .expect("configuration builds");
        let pipeline = sk_build_otlp_pipeline(configuration).expect("OTLP pipeline builds cleanly");
        let _ = pipeline.shutdown();
    }
}
