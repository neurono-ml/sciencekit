# Proposal

## Why

The JVM ecosystem (Java, Scala, Clojure) is large and can share one native binding. A JNI
library over `sciencekit_ffi` plus Java classes gives all three languages a common API;
Scala and Clojure wrappers add idiomatic ergonomics without a second native layer.

## What Changes

- New `sciencekit_jvm` Rust crate producing a JNI `cdylib` over `sciencekit_ffi`.
- Java package `com.neurono.sciencekit` with builder/model classes, `fit`, `predict`,
  `transform`, `score`, `save`, `load`, `close` and `AutoCloseable`.
- Array conversion via JNI primitive arrays (copy) and an optional direct `ByteBuffer`
  zero-copy path; shape validation.
- Errors mapped to Java exceptions (`ScienceKitException` subclasses) carrying status and
  message.
- Scala wrapper module and Clojure namespace over the Java API.
- Build packaging: Gradle/Maven artifacts, sbt and deps.edn examples, native library
  loading, CI and tests (JUnit, ScalaTest, clojure.test).
- Explicit non-goals: Java-native estimators, async, GPU, Kotlin-first API.

## Capabilities

### New Capabilities

- `jvm-bindings`: JNI library, Java API, Scala/Clojure wrappers, array conversion, errors,
  lifecycle and packaging.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_jvm` workspace member, Java/Scala/Clojure packages.
- **APIs:** additive.
- **Dependencies:** `jni 0.22.4` (MSRV 1.85) and the JVM toolchain for packaging.
- **Downstream:** exposes the library to Java, Scala and Clojure users.
- **Acceptance criteria:** JNI loads, Java/Scala/Clojure fit/predict, arrays round-trip,
  errors map and lifecycle is safe.
