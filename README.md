# sciencekit

[![CI](https://github.com/neurono-ml/sciencekit/actions/workflows/ci.yml/badge.svg)](https://github.com/neurono-ml/sciencekit/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/sciencekit)](https://crates.io/crates/sciencekit)
[![docs.rs](https://img.shields.io/docsrs/sciencekit)](https://docs.rs/sciencekit)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-pending-lightgrey)

A machine learning library in Rust reimplementing all of scikit-learn — extreme performance, zero-copy APIs and native out-of-core processing.

## Highlights

- Full scikit-learn surface with idiomatic Rust builder-pattern APIs
- Zero-copy public APIs built on `ndarray` views and sparse `sprs` views
- Pluggable execution via `SKExecutionMode`: CPU (rayon), SIMD/BLAS, GPU (OpenCL/CUDA/ROCm), streaming/out-of-core

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data In"]
        A[("ndarray dense")]:::data
        B[("sprs sparse")]:::data
    end
    subgraph Core["sciencekit Core"]
        C["Estimators · Transformers · Metrics"]:::algo
    end
    subgraph Exec["SKExecutionMode"]
        D["CPU rayon"]:::exec
        E["SIMD / BLAS"]:::exec
        F["GPU OpenCL / CUDA / ROCm"]:::exec
        G["Streaming out-of-core"]:::exec
    end
    H["Python Bindings"]:::bind
    I["mdBook Docs"]:::doc
    A --> C
    B --> C
    C --> D
    C --> E
    C --> F
    C --> G
    D --> H
    E --> H
    F --> H
    G --> H
    C --> I
    classDef data fill:#ffd166,stroke:#b07d00,color:#3b2f00
    classDef algo fill:#06d6a0,stroke:#007a58,color:#00301f
    classDef exec fill:#118ab2,stroke:#0b5d78,color:#ffffff
    classDef bind fill:#ef476f,stroke:#9c2e49,color:#ffffff
    classDef doc fill:#8338ec,stroke:#5b1fa8,color:#ffffff
```

## Documentation

The full documentation — API reference, usage examples and a description of every function — lives in the mdBook on the [`docs/documentations`](https://github.com/neurono-ml/sciencekit/tree/docs/documentations) branch, published via GitHub Pages at <https://neurono-ml.github.io/sciencekit/>.

Release notes: [CHANGELOG.md](CHANGELOG.md). Contribution workflow for agents and humans: [AGENTS.md](AGENTS.md).

## Status

Early development — the Cargo workspace lands with Phase 0 of the roadmap ([docs/PRD.md](docs/PRD.md)).
