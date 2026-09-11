# Developer Book

Welcome to the **Developer Book** — a textbook *inside* the sciencekit documentation. The other pages tell you
**what sciencekit is**. These chapters tell you **how sciencekit was built** — decision by
decision, pull request by pull request — and, more importantly, *why* each choice was made,
and what alternatives were deliberately thrown away.

You do not need a background in Rust or in numerical computing to read it. You need:

- **Basic programming** — you know what a function, a loop, a variable and an `if` are.
- **Curiosity** — you want to know *why*, not just *what*.
- **Patience** — some ideas (traits, generics, zero-copy) take a chapter or two to click.

## How to read this book

Each chapter corresponds to one real pull request that changed the code, and each is written
as a self-contained lesson with the same skeleton:

1. **The problem we were solving** — a plain-language story, before any math.
2. **The decisions — and the roads not taken** — the choices, and the alternatives we rejected.
3. **The concepts, taught from zero** — the Rust and numerical ideas the chapter needs.
4. **Each object, explained** — every public `struct`, `trait`, `enum` and function, and *why* it exists.
5. **A real walk-through, using the tests** — the actual tests from the repository, read line by line.
6. **Look inside** — a diagram and a strong insight.
7. **Recap** — what you should take away.

## Why we use the tests as examples

sciencekit is built with test-driven development: the tests in the `*_tests.rs` companion
modules are the *executable* specification of every function. When a chapter says "here is
how this works", it points you at the real test that proves it. You can read the code, then
run `cargo test` and watch it pass. The test is the proof; the chapter is the explanation.

## The order matters

The chapters follow the order the code was written. Earlier chapters build concepts that
later chapters assume. In particular, **Core contracts** is the keystone — almost
every later chapter builds on the types it introduces. If you only read one chapter first,
read that one.

```
Bootstrap: workspace & CI     — the skeleton
Core contracts                — the vocabulary (traits, errors, views)  ← keystone
Math kernels & BLAS           — the numeric substrate
Execution & observability
The host-centric backend
Module hygiene                — how the code is organised as it grows
Parallel kernels & streaming
```

## How to keep this book honest

sciencekit is still being built. When new algorithms land, or a bug fix changes behaviour,
these chapters must change too. The final chapter, [Maintaining this book](./maintaining.html),
explains that rule: new features get a **new** chapter; fixes and refactors **update** an
existing one. If you find a chapter that no longer matches the code, that is a bug — in the
book, not in the code — and it should be fixed the same way a code bug is.