# Spec Delta

## Purpose

Defines the CUDA backend: device discovery, buffers, kernel compilation and dispatch, CPU
fallback and numerical parity with the CPU kernels.

## ADDED Requirements

### Requirement: CUDA device discovery

The backend SHALL enumerate CUDA devices through the driver API, expose their
kind/name/memory/compute capability and select a device deterministically.

#### Scenario: Device list

- **WHEN** the backend enumerates devices
- **THEN** it returns a stable order with at least one compatible device when available

#### Scenario: No driver

- **WHEN** no CUDA driver is installed
- **THEN** enumeration returns an empty list and the backend reports CPU fallback

### Requirement: CUDA buffers

The backend SHALL allocate device buffers, upload host views, download results and reject
shape/dtype/capacity mismatches.

#### Scenario: Round-trip

- **WHEN** a host matrix is uploaded and downloaded
- **THEN** the values match within the documented tolerance

#### Scenario: Transfer failure

- **WHEN** a copy fails
- **THEN** the backend returns a structured error and never panics

### Requirement: CUDA kernels

The backend SHALL implement pairwise distance, GEMM, elementwise and tree-prediction
kernels, compiling/caching them per device and compute capability.

#### Scenario: Pairwise parity

- **WHEN** the pairwise kernel runs on a device
- **THEN** the result matches the CPU kernel within the documented tolerance

#### Scenario: Unsupported compute capability

- **WHEN** a kernel cannot compile for the device
- **THEN** the backend reports the build log and falls back to CPU

### Requirement: CPU fallback

When the feature is disabled, the driver/device is missing or a kernel is unsupported, the
backend SHALL fall back to CPU and record the reason.

#### Scenario: Feature disabled

- **WHEN** the `gpu-cuda` feature is off
- **THEN** no CUDA call is made and CPU is used

### Requirement: Concurrency and parity

The backend SHALL be safe under concurrent operations, and every kernel SHALL have a
CPU-parity test on small and large inputs.

#### Scenario: Concurrent dispatch

- **WHEN** several threads dispatch the same kernel
- **THEN** results equal the sequential result

#### Scenario: Large input parity

- **WHEN** a large input runs on GPU and CPU
- **THEN** results agree within the documented tolerance
