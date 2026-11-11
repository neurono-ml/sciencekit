//! Resource axis controlled by successive-halving rounds.

/// The training budget dimension scaled across successive-halving rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKHalvingResource {
    /// Scales the number of training rows drawn for each cross-validation round.
    #[default]
    NumberOfSamples,
}
