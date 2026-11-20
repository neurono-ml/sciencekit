# Spec Delta

## Purpose

Defines the JVM binding: JNI native library, Java API, Scala/Clojure wrappers, array
conversion, errors, lifecycle and packaging.

## ADDED Requirements

### Requirement: Native library loading

The Java package SHALL load the JNI library through `System.loadLibrary` or an explicit
path and expose a version/feature query.

#### Scenario: Load

- **WHEN** a Java program initializes the library
- **THEN** the native library loads and the version is available

### Requirement: Java API

Builders and fitted models SHALL be Java classes with chainable setters, `fit`, `predict`,
`transform`, `score`, `save`, `load`, `close` and `AutoCloseable`.

#### Scenario: Fit and predict

- **WHEN** a Java builder is configured and fitted
- **THEN** predict returns a Java array with the expected values

#### Scenario: Try-with-resources

- **WHEN** a model is used in try-with-resources
- **THEN** its native handle is freed on close

### Requirement: Array conversion

Java primitive arrays SHALL convert to/from `SKFfiArray` with shape validation; a direct
`ByteBuffer` path MAY avoid copying and SHALL be documented.

#### Scenario: Double matrix

- **WHEN** a `double[][]` is passed
- **THEN** the native call reads it correctly and returns a `double[][]`

#### Scenario: Shape mismatch

- **WHEN** the array shape is not rectangular
- **THEN** the wrapper throws a shape exception before the native call

### Requirement: Scala and Clojure wrappers

Scala and Clojure SHALL wrap the Java API with idiomatic functions/classes and the same
semantics.

#### Scenario: Scala wrapper

- **WHEN** Scala code builds and fits a model
- **THEN** it produces the same predictions as the Java API

#### Scenario: Clojure wrapper

- **WHEN** Clojure code calls the wrapper
- **THEN** it produces the same predictions as the Java API

### Requirement: Errors and lifecycle

Native status codes SHALL map to Java exceptions; handles SHALL be freed exactly once and
use-after-close SHALL raise a clear exception.

#### Scenario: Native error

- **WHEN** a native call fails
- **THEN** the Java exception carries the status/message

#### Scenario: Use after close

- **WHEN** a closed model is used
- **THEN** a clear exception is thrown

### Requirement: Packaging and tests

The project SHALL include Gradle/Maven, sbt and deps.edn packaging plus JUnit, ScalaTest and
clojure.test suites.

#### Scenario: Test suites

- **WHEN** the three language suites run
- **THEN** every family's fixture passes
