# sciencekit development container
#
# One image with every toolchain the repository's lanes need:
# Rust 1.85 (the pinned MSRV), Python + maturin/pytest for the bindings lane,
# Go and Julia for the FFI-based bindings, Lua 5.4 for the Lua binding smoke
# test (the crate itself embeds Lua through `mlua` with the `vendored` feature,
# so no system interpreter is required at runtime), and the OpenCL headers for
# the GPU backend.
#
# The Go and Lua toolchains also serve the extracted language bindings, which
# live in their own repositories (`neurono-ml/sciencekit-go` and
# `neurono-ml/sciencekit-lua`) and consume the `sciencekit_ffi` C ABI built
# here.
#
# ROCm is *not* installed here: it needs a kernel driver and a device that only
# the GPU node provides. Run the GPU lane on that node with
# `rocm/dev-ubuntu-22.04` (see docs/src/contribute.md) and pass the device
# through with `--device=/dev/kfd --device=/dev/dri`.

FROM ubuntu:24.04

ARG RUST_VERSION=1.85.1
ARG GO_VERSION=1.23.4
ARG JULIA_VERSION=1.10.5

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_TERM_COLOR=always \
    RUST_BACKTRACE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        gfortran \
        git \
        libfontconfig1-dev \
        libopenblas-dev \
        libssl-dev \
        lua5.4 \
        luarocks \
        ninja-build \
        ocl-icd-opencl-dev \
        opencl-headers \
        clinfo \
        pkg-config \
        protobuf-compiler \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Rust, pinned to the repository MSRV with the two gate components.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
            --component rustfmt --component clippy
ENV PATH="/root/.cargo/bin:${PATH}"

# Go, for the `go-bindings` lane.
RUN curl -sSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
        | tar -C /usr/local -xz \
    && ln -s /usr/local/go/bin/go /usr/local/bin/go \
    && ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt

# Julia, for the `julia-dataset-interop` lane.
RUN curl -sSL "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
        | tar -C /opt -xz \
    && ln -s "/opt/julia-${JULIA_VERSION}/bin/julia" /usr/local/bin/julia

# Python bindings: maturin builds the PyO3 wheel, pytest runs the suite.
RUN pip3 install --break-system-packages --no-cache-dir maturin pytest numpy

# LuaJIT is not needed; the binding embeds the reference Lua 5.4 VM. `lua5.4`
# above exists only for running `tests/run.lua` directly.

WORKDIR /workspace
