//! Companion tests for [`SKBaggingError`] conversions and formatting.

use std::error::Error;
use std::fmt;

use sciencekit_common::SKError;

use super::{SKBaggingError, SKBaseEstimatorError};

#[derive(Debug, Clone, PartialEq, Eq)]
struct CustomEstimatorError(&'static str);

impl fmt::Display for CustomEstimatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for CustomEstimatorError {}

#[test]
fn central_sk_error_converts_into_central_variant() {
    let central = SKError::EmptyInput { operation: "fit" };
    let converted: SKBaggingError<CustomEstimatorError> = SKBaggingError::from(central);
    assert!(matches!(
        converted,
        SKBaggingError::Central(SKError::EmptyInput { operation: "fit" })
    ));
    assert!(converted.to_string().contains("fit"));
    assert!(converted.source().is_some());
}

#[test]
fn base_error_converts_into_base_variant_preserving_detail() {
    let via_newtype: SKBaggingError<CustomEstimatorError> = SKBaggingError::from(
        SKBaseEstimatorError(CustomEstimatorError("custom rejection")),
    );
    assert!(matches!(
        via_newtype,
        SKBaggingError::Base(CustomEstimatorError("custom rejection"))
    ));

    let via_constructor = SKBaggingError::from_base(CustomEstimatorError("dense only"));
    assert_eq!(
        via_constructor.as_base(),
        Some(&CustomEstimatorError("dense only"))
    );
    assert!(via_constructor.as_central().is_none());
    assert_eq!(via_constructor.to_string(), "dense only");
    assert!(via_constructor.source().is_some());
}

#[test]
fn bagging_error_over_central_sk_error_converts_back_to_sk_error() {
    let central_wrapped: SKBaggingError<SKError> =
        SKBaggingError::from(SKError::EmptyInput { operation: "fit" });
    let unpacked_central: SKError = central_wrapped.into();
    assert!(matches!(
        unpacked_central,
        SKError::EmptyInput { operation: "fit" }
    ));

    let base_wrapped: SKBaggingError<SKError> =
        SKBaggingError::from_base(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "convert to dense",
        });
    let unpacked_base: SKError = base_wrapped.into();
    assert!(matches!(
        unpacked_base,
        SKError::UnsupportedRepresentation {
            representation: "csr",
            ..
        }
    ));
}
