# Design

## Context

See `proposal.md` — Why. `ffi-core` defines the C ABI; the JVM binding can either wrap it
through JNI or call it via the Java FFM API. JNI (`jni 0.22.4`, MSRV 1.85) is the
compatibility baseline; Java classes expose the API, and Scala/Clojure wrap Java.

## Goals / Non-Goals

**Goals:**
- One native binding for Java, Scala and Clojure.
- Safe handle lifecycle and array conversion.
- Build artifacts for the three ecosystems.

**Non-Goals:**
- Java-native estimators, async, GPU, Kotlin-first API.
- Replacing the C ABI with a second native surface.

## Decisions

### Decision 1 — JNI `cdylib` over `sciencekit_ffi`

`sciencekit_jvm` exposes `Java_com_neurono_sciencekit_*` functions, calls the C ABI and
converts Java arrays/handles. A future FFM path can call the C ABI directly without
changing the Java API.

### Decision 2 — Java API

Classes: `ScienceKit`, `Builder`, `Model`, `Splitter`, `ScienceKitException`. Models
implement `AutoCloseable`; builders are mutable; setters return `this`. `close` frees the
handle exactly once; use-after-close raises `IllegalStateException`.

### Decision 3 — Arrays

`double[][]`/`float[][]` are copied into contiguous native buffers for the call; outputs are
copied back. A direct `ByteBuffer` path is optional and documented. Shape validation happens
in Java before the native call.

### Decision 4 — Errors

`SKFfiStatus` maps to `ScienceKitException` subclasses (`ShapeException`,
`InvalidArgumentException`, `UnsupportedException`, `NotFittedException`, `InternalException`)
with status/message.

### Decision 5 — Scala/Clojure wrappers

Scala provides a thin object/class layer (`ScienceKitScala`) and Clojure a namespace
(`sciencekit.core`) over the Java API. They do not add native code.

### Decision 6 — Packaging

Gradle/Maven publish the Java jar plus the native library; sbt/deps.edn consume the jar.
Native library loading supports `java.library.path` and an explicit path. CI runs JUnit,
ScalaTest and clojure.test against the built library.

### Decision 7 — File layout

```
crates/sciencekit_jvm/
  Cargo.toml (cdylib)
  src/{lib.rs, handle.rs, arrays.rs, errors.rs}
java/
  build.gradle.kts, src/main/java/com/neurono/sciencekit/*.java, src/test/java/...
scala/
  build.sbt, src/main/scala/..., src/test/scala/...
clojure/
  deps.edn, src/sciencekit/core.clj, test/sciencekit/core_test.clj
```

## Risks / Trade-offs

- **JNI complexity** → a small function set over the C ABI; safety comments and tests.
- **Array copies** → direct buffers optional; documented.
- **Native library packaging** → Gradle/Maven artifacts and a load path.
- **Three ecosystems** → one Java API and thin wrappers; tests per language.

## Migration Plan

Additive: new crate and JVM packages. No existing API changes.

## Open Questions

None blocking. A Java FFM path is a future compatibility improvement.
