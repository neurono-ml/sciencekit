//! TDD tests for successive-halving candidate and resource schedules
//! (spec `halving-grid-search`, requirement `Successive halving loop`).

use super::{
    SKHalvingResource, scheduler::sk_surviving_candidate_count, sk_halving_candidate_schedule,
    sk_halving_resource_schedule,
};

#[test]
fn halving_scheduler_nine_candidates_shrink_nine_three_one() {
    assert_eq!(sk_halving_candidate_schedule(9, 3), vec![9, 3, 1]);
}

#[test]
fn halving_scheduler_non_multiple_candidate_counts_use_ceiling_division() {
    assert_eq!(sk_halving_candidate_schedule(10, 3), vec![10, 4, 2, 1]);
    assert_eq!(sk_halving_candidate_schedule(1, 3), vec![1]);
}

#[test]
fn halving_scheduler_resource_grows_geometrically_up_to_maximum() {
    assert_eq!(sk_halving_resource_schedule(10, 100, 3), vec![10, 30, 90]);
    assert_eq!(
        sk_halving_resource_schedule(10, 270, 3),
        vec![10, 30, 90, 270]
    );
}

#[test]
fn halving_scheduler_handles_overflow_and_boundary_resources_cleanly() {
    assert_eq!(sk_halving_resource_schedule(50, 50, 3), vec![50]);
    assert_eq!(
        sk_halving_resource_schedule(usize::MAX / 2, usize::MAX, 3),
        vec![usize::MAX / 2]
    );
    assert_eq!(
        SKHalvingResource::default(),
        SKHalvingResource::NumberOfSamples
    );
    assert_eq!(sk_surviving_candidate_count(9, 3, 10, 90, false), 3);
    assert!(sk_halving_resource_schedule(0, 100, 3).is_empty());
    assert!(sk_halving_resource_schedule(50, 10, 3).is_empty());
    assert!(sk_halving_resource_schedule(10, 100, 1).is_empty());
    assert!(sk_halving_candidate_schedule(0, 3).is_empty());
    assert!(sk_halving_candidate_schedule(9, 1).is_empty());
}
