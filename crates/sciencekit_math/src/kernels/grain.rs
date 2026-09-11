//! Measured per-kernel grain constants (spec `higher-order-kernels`).
//!
//! A kernel's **grain** is the minimum number of elements per work unit needed
//! to amortize parallel-dispatch overhead; the resolution rule derives
//! `parallelism = min(cores, ceil(unit / grain))`, so a unit below one grain
//! stays sequential. The same grain applies to in-memory and streaming regimes.
//!
//! # Calibration protocol
//!
//! `cargo bench -p sciencekit_math --bench kernel_calibration` runs each kernel
//! across cheap/medium/expensive closures and sizes 10¹–10⁷ for both the
//! sequential (`parallelism = 1`) and parallel (`parallelism = cores`) plans.
//! The crossover is the smallest size at which the parallel plan beats the
//! sequential one; the recorded grain is the **largest crossover across closure
//! costs** (a conservative, large grain that delays parallelization of heavy
//! closures rather than parallelizing cheap ones too early).
//!
//! # Provenance
//!
//! - Protocol: `kernel_calibration` (criterion 0.5, reduced sample budget).
//! - Machine: 12th Gen Intel(R) Core(TM) i7-12700H, 20 logical cores.
//! - Platform: Linux 7.0.0-31-generic x86_64.
//! - Date: 2026-09-11.
//! - Raw report: `temporary/2026-09-11/parallel-kernel-execution/raw-calibration.txt`.
//!
//! Re-calibration on another machine changes only these constants and this
//! provenance; the kernel behavior contracts are unaffected.

/// Grain for `sk_elementwise_transform`: crossover 10⁶ (cheap closure) on the
/// calibration machine.
pub const SK_ELEMENTWISE_TRANSFORM_GRAIN: usize = 1_000_000;

/// Grain for `sk_binary_combine`: no crossover observed for the cheap closure up
/// to 10⁷ (three-array `par_azip!` overhead dominates), so the conservative
/// grain is set at the largest measured size.
pub const SK_BINARY_COMBINE_GRAIN: usize = 10_000_000;

/// Grain for `sk_axis_sum`: crossover 10⁵ for both axis-0 and axis-1.
pub const SK_AXIS_SUM_GRAIN: usize = 100_000;

/// Grain for `sk_scale_in_place`: crossover 10⁶.
pub const SK_SCALE_IN_PLACE_GRAIN: usize = 1_000_000;