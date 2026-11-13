//! The engine a tabular conversion originates from (spec
//! `dataset-interop-core`).

use std::fmt;

/// The concrete engine behind a tabular source, recorded on conversion spans.
///
/// The neutral core never depends on an engine; the kind is a diagnostic label
/// an adapter supplies so a trace distinguishes an in-memory table from an
/// Arrow, Polars or DataFusion stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SKInteropSourceKind {
    /// Batches already held in memory.
    InMemory,
    /// An Arrow record-batch stream.
    Arrow,
    /// A Polars data frame or batch iterator.
    Polars,
    /// A DataFusion execution stream.
    DataFusion,
    /// A memory-mapped file.
    MemoryMap,
}

impl SKInteropSourceKind {
    /// The stable, lower-case kind name recorded in traces.
    pub fn name(&self) -> &'static str {
        match self {
            SKInteropSourceKind::InMemory => "in_memory",
            SKInteropSourceKind::Arrow => "arrow",
            SKInteropSourceKind::Polars => "polars",
            SKInteropSourceKind::DataFusion => "datafusion",
            SKInteropSourceKind::MemoryMap => "memory_map",
        }
    }
}

impl fmt::Display for SKInteropSourceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}
