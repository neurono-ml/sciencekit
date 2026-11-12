//! Source-level completeness scan for the streaming coverage audit.
//!
//! Comparing the matrix against another hand-written list would not catch a new
//! estimator. This module walks the sibling crate sources at test time and
//! counts the actual `fn fit_streaming` and `pub fn partial_fit` definitions, so
//! adding a streaming estimator without auditing it fails the suite.

use std::fs;
use std::path::{Path, PathBuf};

use super::matrix::{PARTIAL_FIT_METHODS, STREAMING_ESTIMATORS};

/// A source file that defines a streaming entry point.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct StreamingSource {
    /// The file path relative to the workspace `crates/` directory.
    pub(super) path: String,
    /// The `fit_streaming` definitions in the file.
    pub(super) fit_streaming: usize,
    /// The public `partial_fit` definitions in the file.
    pub(super) partial_fit: usize,
}

/// Walks `crates/` and collects every file that defines a streaming entry point.
pub(super) fn sk_scan_streaming_sources() -> Vec<StreamingSource> {
    let crates_directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the pipeline crate lives under crates/")
        .to_path_buf();
    let mut sources = Vec::new();
    sk_visit(&crates_directory, &crates_directory, &mut sources);
    sources.sort();
    sources
}

fn sk_visit(root: &Path, directory: &Path, sources: &mut Vec<StreamingSource>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path: PathBuf = entry.path();
        if path.is_dir() {
            sk_visit(root, &path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sk_consider(root, &path, sources);
        }
    }
}

fn sk_consider(root: &Path, path: &Path, sources: &mut Vec<StreamingSource>) {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if file_name.contains("tests") || file_name.ends_with("test_datasets.rs") {
        return;
    }
    let Ok(contents) = fs::read_to_string(path) else {
        return;
    };
    let fit_streaming = contents.matches("fn fit_streaming").count();
    let partial_fit = contents.matches("pub fn partial_fit").count();
    if fit_streaming == 0 && partial_fit == 0 {
        return;
    }
    let relative = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    if relative.starts_with("sciencekit_pipeline/src/streaming_coverage/")
        || relative.starts_with("sciencekit_pipeline/src/streaming/")
    {
        // The audit module's own source mentions the method names it searches
        // for, and the streaming-pipeline contract plus its fixtures are
        // pipeline stages rather than estimator streaming entry points; other
        // files in the crate are still scanned.
        return;
    }
    sources.push(StreamingSource {
        path: relative,
        fit_streaming,
        partial_fit,
    });
}

/// The source scan finds exactly the audited number of streaming entry points.
#[test]
fn the_source_scan_matches_the_audited_streaming_surface() {
    let sources = sk_scan_streaming_sources();
    let fit_streaming: usize = sources.iter().map(|source| source.fit_streaming).sum();
    let partial_fit: usize = sources.iter().map(|source| source.partial_fit).sum();
    assert_eq!(
        fit_streaming, STREAMING_ESTIMATORS,
        "a `fit_streaming` definition was added or removed without a matrix row: {sources:?}"
    );
    assert_eq!(
        partial_fit, PARTIAL_FIT_METHODS,
        "a `partial_fit` definition was added or removed without a matrix row: {sources:?}"
    );
}

/// The scan really reads the sibling crates (guards against a vacuous walk).
#[test]
fn the_source_scan_reaches_every_streaming_crate() {
    let sources = sk_scan_streaming_sources();
    for crate_name in [
        "sciencekit_preprocessing/",
        "sciencekit_impute/",
        "sciencekit_linear_model/",
    ] {
        assert!(
            sources
                .iter()
                .any(|source| source.path.starts_with(crate_name)),
            "the scan did not reach {crate_name}"
        );
    }
}
