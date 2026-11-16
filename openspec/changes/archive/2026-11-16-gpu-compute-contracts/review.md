# Review — gpu-compute-contracts

Reviewer: independent review pass over the `gpu-compute-contracts` spec, the
`sciencekit_gpu` / `sciencekit_common` implementation and the companion tests. The code and
the openspec plan are merged; this review closes task 4.3.

## Scenario coverage

| Requirement / scenario | Evidence |
| --- | --- |
| GPU kind is recorded in the span | `sciencekit_common::observability::gpu_span_tests::the_selected_device_kind_is_recorded_in_the_span`, `..._host_backend_kind_is_recorded_unchanged_without_fallback` |
| CPU fallback records the reason | `gpu_span_tests::unregistered_device_kind_falls_back_to_host_and_records_reason`, `..._the_fallback_reason_is_recorded_once_on_the_span`, `gpu-opencl` `fallback_tests::an_unavailable_operation_records_the_host_fallback_and_reason` |
| Backend-kind vocabulary | `observability::backend_kind_tests::display_names_are_the_documented_spellings`, `..._parsing_rejects_an_unknown_name`, `..._device_kinds_are_distinguished_from_host_kinds`, `..._the_default_kind_is_the_pure_rust_host_backend` |
| Device identity, memory and capabilities | `device::device_tests::device_info_reports_identity_memory_and_capabilities`, `..._scalar_type_support_follows_the_capability_flags`, `..._the_default_capabilities_support_nothing` |
| Multiple devices are ordered | `device::selection_tests::multiple_devices_are_ordered_by_kind_priority_then_index`, `..._selection_is_deterministic_and_honours_the_device_index` |
| No device returns an empty list without error | `device::device_tests::an_empty_inventory_has_no_device_to_select`, `opencl::device::device_tests::an_empty_inventory_has_no_device_to_select` |
| Buffer round-trip within tolerance | `buffer::buffer_tests::a_host_buffer_round_trips_through_a_mock_device`, `..._single_precision_buffers_round_trip`, OpenCL `buffer::buffer_tests::buffers_round_trip_without_loss` |
| Shape mismatch is rejected | `buffer::buffer_validation_tests::an_upload_exceeding_the_device_capacity_returns_shape_mismatch`, `..._a_shape_exceeding_the_buffer_capacity_is_rejected`, `..._downloading_a_buffer_from_another_backend_is_rejected` |
| Supported kernel dispatches on the device | `kernel::kernel_tests::a_supported_kernel_dispatches_and_returns_a_buffer`, `..._the_kernel_catalogue_lists_every_kernel` |
| Unsupported kernel names itself | `kernel::kernel_tests::an_unsupported_kernel_names_itself_in_the_error` |
| Automatic planning selects the GPU | `execution::gpu_planning_tests::automatic_plan_selects_the_configured_device_when_available`, `..._automatic_plan_skips_incompatible_first_device_for_capable_second_device` |
| Automatic planning falls back deterministically | `gpu_planning_tests::automatic_plan_falls_back_to_host_without_a_device`, `..._falls_back_when_the_kernel_is_missing`, `..._falls_back_when_the_scalar_type_is_unsupported`, `..._keeps_the_configured_host_backend`, `..._gpu_planning_is_deterministic` |
| Concurrent discovery/planning/transfers | `backend::concurrency_tests::concurrent_discovery_agrees`, `..._concurrent_planning_agrees`, `..._concurrent_transfers_agree`, OpenCL `concurrency_tests::concurrent_pairwise_dispatches_agree`, `..._concurrent_transfers_agree` |
| Structured transfer failures, no panic across FFI | `buffer::buffer_validation_tests::uploading_to_a_backend_without_devices_reports_the_device_unavailable`, `..._a_host_backend_cannot_own_a_device_buffer`, OpenCL `fallback_tests::fallback_operations_report_structured_errors` |

## Findings

1. **Minor — hardware-gated scenarios.** The OpenCL scenarios that need a real device are
   skipped when no device is present; the mock inventory covers the contract surface. This
   is the documented GPU gate in `dag.yaml` (`policy` guardrails), not a spec gap.
2. **Minor — CUDA/ROCm.** Both kinds are part of the vocabulary; their backends are
   deferred to community hardware (ADR #408). The contract tests cover the kinds, the
   planning paths and the fallback, which is what this change owns.

## Verdict

PASS — every scenario in the spec has a proving test on the contract surface, and the
fallback path is explicit and deterministic.
