# Tasks — openspec-config-rules-quoting

Tooling-only change (`skip_specs: true`). Verification replaces the TDD red/green pair: the
"red" is the warning the CLI prints today, the "green" is its absence with the rules visible.

## 1. Reproduce and fix

- [ ] 1.1 Red: capture the current behavior — run `openspec status --change "<any change>"`,
  `openspec instructions proposal --change "<change>" --json` and
  `openspec instructions apply --change "<change>" --json`, and record the
  `Rules for … ignoring` and `Guidance for operation … ignoring` warnings plus the fact that
  the rules are absent from the JSON. *(commit `chore(openspec): capture the discarded config rules`)*
- [ ] 1.2 Rewrite every `rules.proposal`, `rules.tasks` and `operations.apply.guidance`
  item in `openspec/config.yaml` as a folded block scalar (`>-`), preserving the text
  verbatim. *(commit `chore(openspec): quote the config rules as folded scalars`)*
- [ ] 1.3 Green: rerun the same three commands and assert no warning is printed and that a
  distinctive phrase from each rule list and from the apply guidance appears in the JSON
  output. *(commit `chore(openspec): verify the config rules reach the artifact instructions`)*

## 2. Gates

- [ ] 2.1 `openspec validate --strict` passes for both open changes
  (`sgd-classifier-regressor`, `streaming-source-provider`) with no config warning.
