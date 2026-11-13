<!-- DEFERRED (2026-11-13): the JVM lane is deferred by ADR #381 — there is no
`sciencekit-jvm` repository, no JNI layer and no owner. Keep the plan, ship no JVM code. -->

# Tasks

## 1. JNI library

- [ ] 1.1 DEFERRED (2026-11-13): Create `crates/sciencekit_jvm` with a `cdylib` and JNI functions for version/
  status/handles; verify with a JNI load smoke test.
- [ ] 1.2 DEFERRED (2026-11-13): Implement Java array↔native conversion and shape validation; verify with a JUnit
  test.
- [ ] 1.3 DEFERRED (2026-11-13): Implement builder/model JNI functions and `AutoCloseable`/use-after-close; verify
  with JUnit tests.

## 2. Java API

- [ ] 2.1 DEFERRED (2026-11-13): Implement the Java builder/model classes and exception hierarchy; verify with
  JUnit fit/predict fixtures.
- [ ] 2.2 DEFERRED (2026-11-13): Implement save/load and try-with-resources; verify with JUnit tests.
- [ ] 2.3 DEFERRED (2026-11-13): Add concurrency and large-array tests; verify with JUnit tests.

## 3. Scala and Clojure wrappers

- [ ] 3.1 DEFERRED (2026-11-13): Implement the Scala wrapper and fixtures; verify with ScalaTest.
- [ ] 3.2 DEFERRED (2026-11-13): Implement the Clojure namespace and fixtures; verify with clojure.test.
- [ ] 3.3 DEFERRED (2026-11-13): Add packaging (Gradle/Maven, sbt, deps.edn) and native library loading tests;
  verify with a clean build.

## 4. Acceptance and review

- [ ] 4.1 DEFERRED (2026-11-13): Run the three language suites and a CI matrix against the built library; verify
  all pass.
- [ ] 4.2 DEFERRED (2026-11-13): Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 DEFERRED (2026-11-13): Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` subsection on JNI handles and array copies); verify with
  `mdbook build docs`.
- [ ] 4.4 DEFERRED (2026-11-13): Independent review by a separate agent validating every scenario in
  `jvm-bindings`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
