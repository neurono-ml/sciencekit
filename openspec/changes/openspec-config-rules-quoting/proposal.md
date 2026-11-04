## Why

Every `openspec` command prints:

```
Rules for 'proposal' must be an array of strings, ignoring this artifact's rules.
Rules for 'tasks' must be an array of strings, ignoring this artifact's rules.
Guidance for operation 'apply' must be an array of strings, ignoring this operation's guidance.
```

The items in `openspec/config.yaml` contain `": "`, so YAML parses them as
key/value mappings instead of strings. The project's rules for `proposal` and `tasks` and the
whole `apply` guidance are therefore **silently discarded** — artifact instructions no
longer carry the TDD, acceptance-criteria, backend-selection, worktree or CI-gate rules that
AGENTS.md promises. This change restores them.

## What Changes

- Rewrite every `rules.<artifact>` item and every `operations.apply.guidance` item in
  `openspec/config.yaml` as a folded block scalar (`>-`), so a stray `": "` can never turn a
  rule into a mapping again.
- Keep the rule text verbatim: this is a transport fix, not a policy change.
- Verify with `openspec instructions proposal --change <change> --json` and
  `openspec instructions tasks --change <change> --json` that the rules appear in the output
  and that no warning is printed; likewise for `openspec instructions apply`.
- No specification changes: this is tooling configuration, not library behavior.

## Capabilities

### New Capabilities

_None._ Tooling-only change (`skip_specs: true`).

### Modified Capabilities

_None._

## Impact

- **Affected file:** `openspec/config.yaml` only.
- **Affected behavior:** artifact instructions emitted by the `openspec` CLI, and the apply
  guidance consumed by agents.
- **No code, no public API, no dependencies.**
- **Acceptance:** no `Rules for … were ignored` or `Guidance for operation … was ignored`
  warning is printed by `openspec status`, `openspec instructions` or `openspec validate`.
