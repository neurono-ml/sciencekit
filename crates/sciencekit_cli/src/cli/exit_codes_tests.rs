//! Unit tests for the stable exit-code contract.

use super::SKExitCode;

#[test]
fn every_exit_code_keeps_its_documented_number() {
    assert_eq!(SKExitCode::Success.code(), 0);
    assert_eq!(SKExitCode::Runtime.code(), 1);
    assert_eq!(SKExitCode::Usage.code(), 2);
    assert_eq!(SKExitCode::Data.code(), 3);
    assert_eq!(SKExitCode::UnsupportedFeature.code(), 4);
}
