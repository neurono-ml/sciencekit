# Design

## Context

See `proposal.md` — Why. `sciencekit_linear_model` already has a correct
`pub(crate) SKRandomGenerator` (SplitMix64 + rejection sampling + Fisher-Yates). The
library needs it in `sciencekit_common` so every crate can depend on one implementation
without duplicating the constants or the sampling policy.

## Goals / Non-Goals

**Goals:**
- One seeded generator with golden-vector tests.
- One sub-seed derivation rule.
- A documented reproducibility scope.
- Refactor the SGD code without changing its seeded behavior.

**Non-Goals:**
- Cryptographic randomness.
- A global or thread-local RNG.
- Parallel streams that share mutable state (each task owns its generator).
- Bit-for-bit reproducibility across platforms/backends.

## Decisions

### Decision 1 — Promote SplitMix64 to `sciencekit_common`

The generator moves unchanged (same constants, rejection sampling and Fisher-Yates) to
`sciencekit_common::random::SKRandomGenerator`. Methods become `pub`; the module is
re-exported from the crate root. The linear-model SGD code imports it and deletes its
private copy.

Alternative considered: a new external RNG crate (adds a dependency and a second seed
convention); a per-crate copy (drifts and duplicates the golden vectors).

### Decision 2 — Add float and normal draws

The current generator exposes integers, indices and shuffles. The shared version adds:

- `next_f64()`: uniform in `[0, 1)` from the top 53 bits;
- `uniform(low, high)`: scaled `next_f64`;
- `normal()`: Box-Muller with a cached spare, so two draws consume one uniform pair;
- `sample_indices_without_replacement(n, k)`: partial Fisher-Yates.

All draws use the same underlying `next_u64` stream; no platform libm is required for the
normal (use `sqrt`/`ln`/`cos` from `std`; values are deterministic for a given
implementation).

### Decision 3 — Sub-seed derivation

`sk_derive_seed(global, domain, index)` mixes the three values with SplitMix64 constants and
a domain tag:

```rust
pub fn sk_derive_seed(global_seed: u64, domain: u64, index: u64) -> u64
```

Domains are named constants (`SK_RANDOM_DOMAIN_TREE`, `..._INITIALIZATION`, `..._STAGE`,
`..._FOLD`, `..._PROJECTION`). The function is pure and order-independent, so a component
gets the same seed regardless of when or on which thread it runs.

### Decision 4 — Ownership and thread safety

The generator is `Send`; because every draw takes `&mut self`, sharing one between
threads is a compile-time impossibility without external synchronization, so each
component owns its own. Each task owns its own
generator; no `Mutex` is needed. The only shared state is the immutable global seed and the
domain/index values.

### Decision 5 — Reproducibility scope

Documentation states: same library version + seed + input + backend → identical result;
thread-count invariance is tested; cross-platform/cross-backend differences are tolerated
where floating-point reduction order differs. The generator itself is bit-deterministic
because it uses only integer arithmetic and IEEE-754 basics.

### Decision 6 — Test strategy

- Golden vectors for `next_u64` at several seeds.
- Range and permutation tests for `next_f64`, `uniform`, `shuffle` and
  `sample_indices_without_replacement`.
- Normal moments (mean ~0, variance ~1) over a large sample, with a documented tolerance.
- Sub-seed distinctness and order-independence.
- Linear-model SGD tests remain green with the shared generator (same seeded behavior).

### Decision 7 — File layout

```
crates/sciencekit_common/src/random/
  mod.rs
  generator.rs
  sub_seed.rs
  *_tests.rs
crates/sciencekit_linear_model/src/sgd/shuffle.rs   (deleted; imports the shared type)
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Box-Muller uses transcendental functions** → the same build/platform is deterministic;
  cross-platform bit-identity is not promised.
- **`Send`, `&mut self` draw** → tasks own generators; sharing requires external
  synchronization, which the library does not need.
- **Refactor touches SGD internals** → tests already pin seeded behavior; run the full
  linear-model suite.
- **Domain tags are a convention** → named constants and a sub-seed test prevent accidental
  reuse.

## Migration Plan

Additive for common; internal refactor for linear-model. Reverting restores the private
generator.

## Open Questions

None blocking. A parallel-stream API is a future convenience.
