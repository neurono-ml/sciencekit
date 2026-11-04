//! The learning-rate selection value (spec `stochastic-gradient-solver`,
//! decision 4).
//!
//! The selection is configuration; the schedules that consume it live in
//! `schedule.rs` and `optimal.rs`.

/// The learning-rate schedule of a stochastic-gradient fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKSGDLearningRate {
    /// `η_t = η₀`, a fixed rate.
    Constant,
    /// `η_t = 1 / (alpha · (t + t₀))`, estimated from the loss scale.
    Optimal,
    /// `η_t = η₀ / (1 + t)^power`, a decaying rate.
    InverseScaling,
    /// `η₀`, halved after every non-improving epoch.
    Adaptive,
}

impl SKSGDLearningRate {
    /// Whether the schedule consults the configured initial learning rate.
    pub(crate) fn uses_initial_rate(self) -> bool {
        !matches!(self, Self::Optimal)
    }
}
