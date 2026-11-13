//! Builder-only growth constraints for [`super::SKTreeStructure`].

use sciencekit_common::{SKError, SKFloat, sk_validate_hyperparameter};

use crate::splitter::{SKMaximumFeatures, SKSplitter};

/// Growth constraints passed to [`super::SKTreeStructure::build_structure`].
#[derive(Debug, Clone)]
pub struct SKStructureConstraints<F> {
    pub(super) maximum_depth: Option<usize>,
    pub(super) minimum_samples_split: usize,
    pub(super) minimum_samples_leaf: usize,
    pub(super) maximum_features: SKMaximumFeatures<F>,
    pub(super) splitter: SKSplitter,
    pub(super) random_seed: u64,
}

impl<F: SKFloat> SKStructureConstraints<F> {
    /// Start building structure constraints (the only construction surface).
    pub fn build() -> SKStructureConstraintsBuilder<F> {
        SKStructureConstraintsBuilder::new()
    }

    /// Maximum depth, or `None` when unbounded by depth.
    pub fn maximum_depth(&self) -> Option<usize> {
        self.maximum_depth
    }

    /// Minimum number of samples required to split an internal node.
    pub fn minimum_samples_split(&self) -> usize {
        self.minimum_samples_split
    }

    /// Minimum number of samples required in each child leaf.
    pub fn minimum_samples_leaf(&self) -> usize {
        self.minimum_samples_leaf
    }

    /// Feature-subsampling policy applied at each split.
    pub fn maximum_features(&self) -> &SKMaximumFeatures<F> {
        &self.maximum_features
    }

    /// Candidate threshold strategy (`Best` or `Random`).
    pub fn splitter(&self) -> SKSplitter {
        self.splitter
    }

    /// Deterministic seed used for stochastic feature/split selection.
    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }
}

/// Builder for [`SKStructureConstraints`].
#[derive(Debug, Clone)]
pub struct SKStructureConstraintsBuilder<F> {
    maximum_depth: Option<usize>,
    minimum_samples_split: usize,
    minimum_samples_leaf: usize,
    maximum_features: SKMaximumFeatures<F>,
    splitter: SKSplitter,
    random_seed: u64,
}

impl<F: SKFloat> SKStructureConstraintsBuilder<F> {
    /// Create a fresh builder with CART defaults (`maximum_depth = None`,
    /// `minimum_samples_split = 2`, `minimum_samples_leaf = 1`, `All`, `Best`,
    /// `random_seed = 0`).
    pub fn new() -> Self {
        Self {
            maximum_depth: None,
            minimum_samples_split: 2,
            minimum_samples_leaf: 1,
            maximum_features: SKMaximumFeatures::All,
            splitter: SKSplitter::Best,
            random_seed: 0,
        }
    }

    /// Stop growing beyond `maximum_depth` edges from the root.
    pub fn maximum_depth(mut self, maximum_depth: usize) -> Self {
        self.maximum_depth = Some(maximum_depth);
        self
    }

    /// Require at least `minimum_samples_split` samples to split a node.
    pub fn minimum_samples_split(mut self, minimum_samples_split: usize) -> Self {
        self.minimum_samples_split = minimum_samples_split;
        self
    }

    /// Require at least `minimum_samples_leaf` samples in every child leaf.
    pub fn minimum_samples_leaf(mut self, minimum_samples_leaf: usize) -> Self {
        self.minimum_samples_leaf = minimum_samples_leaf;
        self
    }

    /// Configure how many features each split inspects.
    pub fn maximum_features(mut self, maximum_features: SKMaximumFeatures<F>) -> Self {
        self.maximum_features = maximum_features;
        self
    }

    /// Configure best vs random threshold selection.
    pub fn splitter(mut self, splitter: SKSplitter) -> Self {
        self.splitter = splitter;
        self
    }

    /// Configure the deterministic random seed.
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = random_seed;
        self
    }

    /// Validate and construct [`SKStructureConstraints`].
    pub fn build(self) -> Result<SKStructureConstraints<F>, SKError> {
        sk_validate_hyperparameter(
            "minimum_samples_split",
            self.minimum_samples_split >= 2,
            "minimum_samples_split must be at least 2",
        )?;
        sk_validate_hyperparameter(
            "minimum_samples_leaf",
            self.minimum_samples_leaf >= 1,
            "minimum_samples_leaf must be at least 1",
        )?;
        match &self.maximum_features {
            SKMaximumFeatures::Count(count) => sk_validate_hyperparameter(
                "maximum_features",
                *count >= 1,
                "an explicit feature count must be at least 1",
            )?,
            SKMaximumFeatures::Fraction(fraction) => sk_validate_hyperparameter(
                "maximum_features",
                fraction.is_finite() && *fraction > F::zero() && *fraction <= F::one(),
                "a feature fraction must be finite and in (0, 1]",
            )?,
            _ => {}
        }
        Ok(SKStructureConstraints {
            maximum_depth: self.maximum_depth,
            minimum_samples_split: self.minimum_samples_split,
            minimum_samples_leaf: self.minimum_samples_leaf,
            maximum_features: self.maximum_features,
            splitter: self.splitter,
            random_seed: self.random_seed,
        })
    }
}

impl<F: SKFloat> Default for SKStructureConstraintsBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}
