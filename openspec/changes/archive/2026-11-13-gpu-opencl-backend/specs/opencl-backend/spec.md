# Spec Delta

## Purpose

Defines the OpenCL backend: device discovery, buffers, kernel compilation and dispatch,
CPU fallback and numerical parity with the CPU kernels.

## ADDED Requirements

### Requirement: OpenCL device discovery

The backend SHALL enumerate OpenCL platforms and devices through the ICD loader, expose
their kind/name/memory/capabilities and select a device deterministically.

#### Scenario: Device list

- **WHEN** the backend enumerates devices
- **THEN** it returns a stable order with at least one compatible device when available

#### Scenario: No platform

- **WHEN** no OpenCL platform is installed
- **THEN** enumeration returns an empty list and the backend reports CPU fallback

### Requirement: OpenCL buffers

The backend SHALL allocate device buffers, upload host views, download results and reject
shape/dtype/capacity mismatches.

#### Scenario: Round-trip

- **WHEN** a host matrix is uploaded and downloaded
- **THEN** the values match within the documented tolerance

#### Scenario: Out-of-memory

- **WHEN** a buffer exceeds device memory
- **THEN** the backend returns a structured error and falls back where possible

### Requirement: OpenCL kernels

The backend SHALL implement pairwise distance, GEMM, elementwise and tree-prediction
kernels, compiling them once per device and caching the programs.

#### Scenario: Pairwise parity

- **WHEN** the pairwise kernel runs on a device
- **THEN** the result matches the CPU kernel within the documented tolerance

#### Scenario: Kernel compile failure

- **WHEN** a kernel fails to compile on a device
- **THEN** the backend reports the build log and falls back to CPU

### Requirement: CPU fallback

When the feature is disabled, the platform is missing, the device is incompatible or a
kernel is unsupported, the backend SHALL fall back to the CPU path and record the reason.

#### Scenario: Feature disabled

- **WHEN** the `gpu-opencl` feature is off
- **THEN** no OpenCL call is made and CPU is used

#### Scenario: Unsupported kernel

- **WHEN** a requested kernel is not implemented
- **THEN** the operation returns `UnsupportedOperation` and the caller may use CPU

### Requirement: Concurrency and parity

The backend SHALL be safe under concurrent operations, and every kernel SHALL have a
CPU-parity test on small and large inputs.

#### Scenario: Concurrent dispatch

- **WHEN** several threads dispatch the same kernel
- **THEN** results equal the sequential result

#### Scenario: Large input parity

- **WHEN** a large input runs on GPU and CPU
- **THEN** results agree within the documented tolerance
