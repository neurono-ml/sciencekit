//! Bounded in-memory LRU kernel-row cache (spec `support-vector-machines`).

use std::collections::{BTreeMap, HashMap};
use std::mem::size_of;
use std::rc::Rc;

use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_math::SKKernelParameters;

use crate::kernel::{sk_evaluate_kernel_diagonal, sk_evaluate_kernel_row};

/// Bytes per mebibyte.
const BYTES_PER_MEBIBYTE: usize = 1024 * 1024;

/// Bounded LRU cache of on-demand kernel rows keyed by training row index.
#[derive(Debug)]
pub struct SKKernelRowCache<'a, F: SKFloat> {
    features: SKDataView<'a, F>,
    parameters: SKKernelParameters<F>,
    diagonal: Vec<F>,
    max_cached_rows: usize,
    entries: HashMap<usize, (Rc<Vec<F>>, u64)>,
    recency: BTreeMap<u64, usize>,
    next_tick: u64,
    hits: usize,
    misses: usize,
}

impl<'a, F: SKFloat> SKKernelRowCache<'a, F> {
    /// Construct a cache bounded by `cache_size_mib` MiB.
    pub fn from_megabytes(
        features: SKDataView<'a, F>,
        parameters: SKKernelParameters<F>,
        cache_size_mib: usize,
    ) -> Result<Self, SKError> {
        let capacity_bytes = cache_size_mib.saturating_mul(BYTES_PER_MEBIBYTE);
        Self::with_capacity_bytes(features, parameters, capacity_bytes)
    }

    /// Construct a cache bounded by an exact byte capacity.
    pub fn with_capacity_bytes(
        features: SKDataView<'a, F>,
        parameters: SKKernelParameters<F>,
        capacity_bytes: usize,
    ) -> Result<Self, SKError> {
        let diagonal = sk_evaluate_kernel_diagonal(features, &parameters)?;
        let bytes_per_row = diagonal.len().saturating_mul(size_of::<F>());
        let max_cached_rows = if bytes_per_row == 0 {
            0
        } else {
            (capacity_bytes / bytes_per_row).min(diagonal.len())
        };
        Ok(Self {
            features,
            parameters,
            diagonal,
            max_cached_rows,
            entries: HashMap::with_capacity(max_cached_rows),
            recency: BTreeMap::new(),
            next_tick: 0,
            hits: 0,
            misses: 0,
        })
    }

    /// Precomputed kernel diagonal `K(x_i, x_i)`.
    pub fn diagonal(&self) -> &[F] {
        &self.diagonal
    }

    /// Maximum number of rows the cache retains simultaneously.
    pub fn max_cached_rows(&self) -> usize {
        self.max_cached_rows
    }

    /// Number of rows currently resident in the cache.
    pub fn cached_rows_count(&self) -> usize {
        self.entries.len()
    }

    /// Cache hit count.
    pub fn hits(&self) -> usize {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> usize {
        self.misses
    }

    /// Fetch or compute kernel row `row_index`.
    pub fn row(&mut self, row_index: usize) -> Result<Rc<Vec<F>>, SKError> {
        if let Some((row, previous_tick)) = self.entries.get_mut(&row_index) {
            self.recency.remove(previous_tick);
            self.next_tick += 1;
            *previous_tick = self.next_tick;
            self.recency.insert(self.next_tick, row_index);
            self.hits += 1;
            return Ok(Rc::clone(row));
        }

        self.misses += 1;
        let computed = Rc::new(sk_evaluate_kernel_row(
            self.features,
            row_index,
            &self.parameters,
        )?);
        if self.max_cached_rows == 0 {
            return Ok(computed);
        }
        if self.entries.len() >= self.max_cached_rows {
            if let Some((oldest_tick, oldest_row)) = self.recency.pop_first() {
                let _ = oldest_tick;
                self.entries.remove(&oldest_row);
            }
        }
        self.next_tick += 1;
        self.entries
            .insert(row_index, (Rc::clone(&computed), self.next_tick));
        self.recency.insert(self.next_tick, row_index);
        Ok(computed)
    }
}
