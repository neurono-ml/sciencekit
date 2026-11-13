//! Companion TDD tests for the bounded LRU kernel-row cache
//! (spec `support-vector-machines`).

use std::mem::size_of;

use ndarray::array;
use sciencekit_common::SKDataView;
use sciencekit_math::{SKKernel, SKKernelParameters};
use sprs::TriMat;

use super::SKKernelRowCache;

#[test]
fn cache_tracks_hits_misses_lru_eviction_and_recomputation_on_dense_view() {
    let features = array![[1.0_f64, 0.0], [0.0, 2.0], [3.0, 1.0], [-1.0, 2.0]];
    let parameters = SKKernelParameters::new(SKKernel::Linear, 0.0, 1, 0.0);
    // 4 rows => each cached row is 4 * 8 = 32 bytes; allow at most 2 rows (64 bytes).
    let row_bytes = 4 * size_of::<f64>();
    let mut cache = SKKernelRowCache::with_capacity_bytes(
        SKDataView::Dense(features.view()),
        parameters,
        2 * row_bytes,
    )
    .unwrap();

    assert_eq!(cache.maximum_cached_rows(), 2);
    assert_eq!(cache.diagonal(), &[1.0, 4.0, 10.0, 5.0]);

    let r0 = cache.row(0).unwrap();
    assert_eq!(&*r0, &[1.0, 0.0, 3.0, -1.0]);
    assert_eq!(
        (cache.hits(), cache.misses(), cache.cached_rows_count()),
        (0, 1, 1)
    );

    let r1 = cache.row(1).unwrap();
    assert_eq!(&*r1, &[0.0, 4.0, 2.0, 4.0]);
    assert_eq!(
        (cache.hits(), cache.misses(), cache.cached_rows_count()),
        (0, 2, 2)
    );

    // Touch row 0 so row 1 becomes least-recently-used.
    let _ = cache.row(0).unwrap();
    assert_eq!((cache.hits(), cache.misses()), (1, 2));

    // Request row 2: evicts row 1, keeps row 0.
    let r2 = cache.row(2).unwrap();
    assert_eq!(&*r2, &[3.0, 2.0, 10.0, -1.0]);
    assert_eq!(
        (cache.hits(), cache.misses(), cache.cached_rows_count()),
        (1, 3, 2)
    );

    // Row 0 is still resident (hit), while row 1 was evicted and must recompute (miss).
    let _ = cache.row(0).unwrap();
    assert_eq!((cache.hits(), cache.misses()), (2, 3));
    let r1_recomputed = cache.row(1).unwrap();
    assert_eq!(&*r1_recomputed, &[0.0, 4.0, 2.0, 4.0]);
    assert_eq!((cache.hits(), cache.misses()), (2, 4));
}

#[test]
fn cache_supports_csr_views_and_zero_cache_size_recomputes_every_row() {
    let mut triplets = TriMat::new((3, 1_000_000));
    triplets.add_triplet(0, 5, 2.0_f64);
    triplets.add_triplet(1, 5, 3.0);
    triplets.add_triplet(1, 999_999, -1.0);
    triplets.add_triplet(2, 999_999, 4.0);
    let csr = triplets.to_csr();

    let parameters = SKKernelParameters::new(SKKernel::RadialBasisFunction, 0.5, 1, 0.0);
    let mut uncached =
        SKKernelRowCache::from_megabytes(SKDataView::Sparse(csr.view()), parameters, 0).unwrap();
    assert_eq!(uncached.maximum_cached_rows(), 0);
    assert_eq!(uncached.diagonal(), &[1.0, 1.0, 1.0]);

    let first = uncached.row(0).unwrap();
    let second = uncached.row(0).unwrap();
    assert_eq!(first, second);
    assert_eq!(uncached.cached_rows_count(), 0);
    assert_eq!(uncached.hits(), 0);
    assert_eq!(uncached.misses(), 2);
}
