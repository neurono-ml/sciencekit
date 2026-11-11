//! Magic-byte detection, missing-feature guidance and corrupt-stream tests
//! (spec `safetensors-format`).

use std::path::Path;

use sciencekit_common::{SKError, SKModelMetadata, SKModelState, SKTensor, SKTensorDataType};

use super::{
    SK_BROTLI_MAGIC, SK_GZIP_MAGIC, SK_SNAPPY_MAGIC, SKSafetensorsCompression, SKSafetensorsWriter,
    sk_append_safetensors_tensor, sk_detect_safetensors_compression,
    sk_read_model_state_safetensors, sk_read_model_state_safetensors_file,
    sk_write_model_state_safetensors,
};

fn tiny_state() -> SKModelState<f64> {
    let metadata = SKModelMetadata::new("SKRidge", 1, SKTensorDataType::F64, 2).unwrap();
    let weights = SKTensor::floating("ridge.weights", vec![2], vec![1.0_f64, -2.0]).unwrap();
    SKModelState::new(metadata, vec![weights]).unwrap()
}

/// Magic-byte and path-suffix detectors distinguish Gzip, Brotli, Snappy and uncompressed Safetensors.
#[test]
fn safetensors_compression_detects_magic_bytes_and_path_extensions() {
    let mut gzip_sample = SK_GZIP_MAGIC.to_vec();
    gzip_sample.extend_from_slice(&[0x08, 0x00, 0x00]);
    assert_eq!(
        sk_detect_safetensors_compression(&gzip_sample),
        Some(SKSafetensorsCompression::Gzip)
    );

    let mut brotli_sample = SK_BROTLI_MAGIC.to_vec();
    brotli_sample.extend_from_slice(&[0x21, 0x04]);
    assert_eq!(
        sk_detect_safetensors_compression(&brotli_sample),
        Some(SKSafetensorsCompression::Brotli)
    );

    let mut snappy_sample = SK_SNAPPY_MAGIC.to_vec();
    snappy_sample.extend_from_slice(&[0x01, 0x02]);
    assert_eq!(
        sk_detect_safetensors_compression(&snappy_sample),
        Some(SKSafetensorsCompression::Snappy)
    );

    let raw_safetensors = sk_write_model_state_safetensors(&tiny_state()).unwrap();
    assert_eq!(sk_detect_safetensors_compression(&raw_safetensors), None);
    assert_eq!(sk_detect_safetensors_compression(&[]), None);

    assert_eq!(
        SKSafetensorsCompression::from_path(Path::new("m.safetensors.gz")),
        Some(SKSafetensorsCompression::Gzip)
    );
    assert_eq!(
        SKSafetensorsCompression::from_path(Path::new("m.safetensors.brotli")),
        Some(SKSafetensorsCompression::Brotli)
    );
    assert_eq!(
        SKSafetensorsCompression::from_path(Path::new("m.snappy.safetensors")),
        Some(SKSafetensorsCompression::Snappy)
    );
    assert_eq!(
        SKSafetensorsCompression::from_path(Path::new("m.safetensors")),
        None
    );
}

/// Reading, writing or appending compressed payloads without the matching feature returns `UnsupportedOperation` with guidance.
#[test]
fn safetensors_compression_missing_feature_returns_unsupported_operation_with_guidance() {
    let state = tiny_state();

    #[cfg(not(feature = "compression-gzip"))]
    {
        let mut fake_gzip = SK_GZIP_MAGIC.to_vec();
        fake_gzip.extend_from_slice(b"corrupt_or_valid_payload");
        match sk_read_model_state_safetensors::<f64>(&fake_gzip) {
            Err(SKError::UnsupportedOperation { reason, .. }) => {
                assert!(reason.contains("compression-gzip"));
            }
            other => panic!("expected UnsupportedOperation for gzip read, got {other:?}"),
        }
        let writer = SKSafetensorsWriter::builder()
            .compression(SKSafetensorsCompression::Gzip)
            .build()
            .unwrap();
        assert!(matches!(
            writer.write_state_to_bytes(&state),
            Err(SKError::UnsupportedOperation { .. })
        ));
        let temporary_directory = tempfile::tempdir().unwrap();
        let gz_file = temporary_directory.path().join("missing.safetensors.gz");
        std::fs::write(&gz_file, &fake_gzip).unwrap();
        let extra = SKTensor::floating("extra", vec![1], vec![3.0_f64]).unwrap();
        assert!(matches!(
            sk_append_safetensors_tensor(&gz_file, &extra),
            Err(SKError::UnsupportedOperation { .. })
        ));
    }

    #[cfg(not(feature = "compression-brotli"))]
    {
        let mut fake_brotli = SK_BROTLI_MAGIC.to_vec();
        fake_brotli.extend_from_slice(b"payload");
        match sk_read_model_state_safetensors::<f64>(&fake_brotli) {
            Err(SKError::UnsupportedOperation { reason, .. }) => {
                assert!(reason.contains("compression-brotli"));
            }
            other => panic!("expected UnsupportedOperation for brotli read, got {other:?}"),
        }
        let temporary_directory = tempfile::tempdir().unwrap();
        let brotli_path = temporary_directory.path().join("raw.safetensors.brotli");
        std::fs::write(&brotli_path, b"raw_rfc7932_without_magic").unwrap();
        assert!(matches!(
            sk_read_model_state_safetensors_file::<f64>(&brotli_path),
            Err(SKError::UnsupportedOperation { .. })
        ));
    }

    #[cfg(not(feature = "compression-snappy"))]
    {
        let mut fake_snappy = SK_SNAPPY_MAGIC.to_vec();
        fake_snappy.extend_from_slice(b"payload");
        match sk_read_model_state_safetensors::<f64>(&fake_snappy) {
            Err(SKError::UnsupportedOperation { reason, .. }) => {
                assert!(reason.contains("compression-snappy"));
            }
            other => panic!("expected UnsupportedOperation for snappy read, got {other:?}"),
        }
        let _ = &state;
    }
    let _ = (&state, SKSafetensorsWriter::builder().build().unwrap());
    let _ = (
        sk_read_model_state_safetensors::<f64>,
        sk_read_model_state_safetensors_file::<f64>,
        sk_append_safetensors_tensor::<f64>,
    );
}

/// Corrupt compressed payloads return `SKError::Conversion` when their feature is enabled.
#[test]
fn safetensors_compression_corrupt_streams_and_compressed_append_behave_cleanly() {
    #[cfg(feature = "compression-gzip")]
    {
        let mut corrupt_gzip = SK_GZIP_MAGIC.to_vec();
        corrupt_gzip.extend_from_slice(&[0x08, 0xFF, 0xFF, 0x00]);
        assert!(matches!(
            sk_read_model_state_safetensors::<f64>(&corrupt_gzip),
            Err(SKError::Conversion(_))
        ));

        let state = tiny_state();
        let temporary_directory = tempfile::tempdir().unwrap();
        let gz_path = temporary_directory.path().join("appended.safetensors.gz");
        let writer = SKSafetensorsWriter::builder()
            .compression(SKSafetensorsCompression::Gzip)
            .build()
            .unwrap();
        writer.write_state_to_path(&state, &gz_path).unwrap();
        let intercept = SKTensor::floating("ridge.intercept", vec![], vec![0.5_f64]).unwrap();
        writer.append_tensor_to_path(&gz_path, &intercept).unwrap();
        let merged = sk_read_model_state_safetensors_file::<f64>(&gz_path).unwrap();
        assert_eq!(merged.number_of_tensors(), 2);
    }

    #[cfg(feature = "compression-brotli")]
    {
        let mut corrupt_brotli = SK_BROTLI_MAGIC.to_vec();
        corrupt_brotli.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
        assert!(matches!(
            sk_read_model_state_safetensors::<f64>(&corrupt_brotli),
            Err(SKError::Conversion(_))
        ));
    }

    #[cfg(feature = "compression-snappy")]
    {
        let mut corrupt_snappy = SK_SNAPPY_MAGIC.to_vec();
        corrupt_snappy.extend_from_slice(&[0x00, 0x05, 0x00, 0x00, 0xDE, 0xAD]);
        assert!(matches!(
            sk_read_model_state_safetensors::<f64>(&corrupt_snappy),
            Err(SKError::Conversion(_))
        ));
    }
}
