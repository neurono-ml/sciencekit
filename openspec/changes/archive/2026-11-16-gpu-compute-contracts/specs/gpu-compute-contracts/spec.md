# Spec Delta

## Purpose

Defines the device-compute foundation: GPU backend kinds, device discovery, device buffers,
kernel dispatch, GPU-aware execution planning and CPU fallback.

## ADDED Requirements

### Requirement: GPU backend kinds

The library SHALL expose OpenCL, CUDA and ROCm as backend kinds alongside the CPU kinds,
and every operation SHALL record the selected kind in its span.

#### Scenario: GPU kind is recorded

- **WHEN** an operation runs on an OpenCL backend
- **THEN** its span records `opencl`, never a hardcoded CPU value

#### Scenario: CPU fallback is recorded

- **WHEN** a GPU kind is requested but no device is available
- **THEN** the operation falls back to CPU and records the fallback reason

### Requirement: Device discovery

The library SHALL enumerate available devices and expose a deterministic device identity,
kind, memory and capability summary.

#### Scenario: Multiple devices are ordered

- **WHEN** several devices are present
- **THEN** enumeration returns a stable order and the first compatible device is selected

#### Scenario: No device

- **WHEN** no GPU device is present
- **THEN** enumeration returns an empty list without error

### Requirement: Device buffers

The library SHALL allocate device buffers, upload host views and download into host arrays,
validating shape, dtype and capacity.

#### Scenario: Round-trip

- **WHEN** a host buffer is uploaded and downloaded
- **THEN** the values are identical within the documented tolerance

#### Scenario: Shape mismatch

- **WHEN** an upload shape exceeds the buffer capacity
- **THEN** the operation returns `ShapeMismatch`

### Requirement: Kernel dispatch

A compute backend SHALL expose a kernel-dispatch surface for the supported kernels
(pairwise distance, GEMM, elementwise and tree prediction), and SHALL report an
unsupported kernel explicitly.

#### Scenario: Supported kernel

- **WHEN** a backend implements a kernel
- **THEN** dispatch runs it on the device and returns a device buffer

#### Scenario: Unsupported kernel

- **WHEN** a backend does not implement a kernel
- **THEN** dispatch returns `UnsupportedOperation` naming the kernel

### Requirement: GPU-aware planning

Automatic execution planning SHALL select a GPU only when the configured kind, a
compatible device and the required kernel are available; otherwise it SHALL select the CPU
fallback deterministically.

#### Scenario: GPU available

- **WHEN** a compatible device and kernel exist
- **THEN** the plan selects the GPU backend

#### Scenario: GPU unavailable

- **WHEN** the device or kernel is missing
- **THEN** the plan selects CPU and exposes the fallback reason

### Requirement: Concurrency and validation

Device discovery and planning SHALL be safe under concurrent calls, and buffer operations
SHALL reject unavailable devices, unsupported dtypes and transfer failures with structured
errors.

#### Scenario: Concurrent planning agrees

- **WHEN** several threads plan with the same context
- **THEN** every plan is identical

#### Scenario: Transfer failure

- **WHEN** a device transfer fails
- **THEN** the operation returns a structured error and never panics across FFI
