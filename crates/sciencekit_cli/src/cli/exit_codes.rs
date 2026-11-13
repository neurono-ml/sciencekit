//! The stable process exit-code contract (spec `workflows-cli`, requirement
//! "Telemetry and exit codes").

/// The documented exit codes of the `sciencekit` binary.
///
/// A consumer scripts against these numbers: they never change between
/// releases within a major version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKExitCode {
    /// The command completed and every declared artifact was written.
    Success,
    /// An unexpected runtime failure (convergence, numerical, internal).
    Runtime,
    /// Usage or validation failure: bad flags, invalid workflow, bad value.
    Usage,
    /// Dataset or model I/O failure: a missing file, malformed artifact.
    Data,
    /// A capability this build was compiled without.
    UnsupportedFeature,
}

impl SKExitCode {
    /// The numeric exit status.
    pub const fn code(self) -> u8 {
        match self {
            SKExitCode::Success => 0,
            SKExitCode::Runtime => 1,
            SKExitCode::Usage => 2,
            SKExitCode::Data => 3,
            SKExitCode::UnsupportedFeature => 4,
        }
    }
}
