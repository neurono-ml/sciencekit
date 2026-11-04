# Design

## Context

`openspec/config.yaml` stores `rules.<artifact>` and `operations.apply.guidance` as YAML
sequences of plain scalars. A plain scalar cannot contain `": "` (colon followed by a space)
because YAML reads it as a mapping indicator. Several rules do:

```
    - Plan changes in separate groups: (1) algorithm implementation, ...
```

The CLI therefore sees mappings and discards the whole rule list, printing a warning on every
command. The `apply` guidance fails the same way, so agents receive neither the TDD rules nor
the worktree/CI instructions from the config.

## Goals / Non-Goals

**Goals:**

- Restore the rules and guidance without changing a single word of their text.
- Make the failure mode unrepresentable: no future rule can be broken by a `": "`.

**Non-Goals:**

- Changing the rules themselves or adding new ones.
- Adding a CI check for the warning (a separate concern; noted below).

## Decisions

### 1. Folded block scalars for every item

Each rule and guidance item becomes:

```
    - >-
      Plan changes in separate groups: (1) algorithm implementation, ...
```

`>-` folds the indented lines into one space-joined scalar and strips the trailing newline,
which is exactly the current single-line semantics. Any `: ` or `#` inside is literal.

Alternatives considered: **double-quoting** each item — works today, but every future rule
must remember to escape `"`, `\` and newlines; a single omission breaks the list again.
**Literal blocks (`|`)** — preserve newlines that the CLI would then show as broken prose.
**A JSON sidecar** — a second source of truth for the same rules; rejected.

### 2. Verification through the CLI, not a parser test

The contract is what the CLI emits, so the check is: run `openspec status --change <change>`,
`openspec instructions proposal --change <change> --json` and
`openspec instructions apply --change <change> --json`; assert no `ignoring` warning and
assert a known rule phrase appears in the JSON. This catches both parse failures and a
mistaken re-indentation.

## Risks / Trade-offs

- **Re-indentation can silently join or split a rule** → each item is copied verbatim and the
  verification greps a distinctive phrase from the start, middle and end of each list.
- **The warning could be emitted by a different layer** → the reproduction step runs before
  the edit and prints the warning, proving the cause.

## Migration Plan

Single commit to `openspec/config.yaml`. Rollback is reverting the file; no data or API is
involved.
