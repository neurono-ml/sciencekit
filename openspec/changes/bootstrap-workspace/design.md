# Design — bootstrap-workspace

## Context

Repositório sem workspace Cargo, toolchain livre e CI inexistente (ver proposal — Why). Restrições do PRD: MSRV Rust 1.85, edition 2024 (§3.6), licença Apache-2.0 (§11.3), PRs pequenas com gates de qualidade contínuos (§3.2). O workspace nasce vazio de membros; os 19 sub-crates entram um por change.

## Goals / Non-Goals

**Goals:**
- Ambiente de build idêntico em qualquer máquina: toolchain fixado por arquivo versionado, não por convenção.
- Portão de CI único que serve a todos os crates futuros sem edição por crate adicionado (`--workspace` cobre tudo).
- Base limpa para o TDD obrigatório: `cargo test` verde num clone recém-feito.

**Non-Goals:**
- Criar sub-crates vazios em lote (churn sem valor; cada crate nasce na sua change).
- Configurar branch protection no GitHub (ação administrativa fora do repositório).
- Perfis de release, `cargo-deny`, cobertura, benchmarks — chegam quando houver código que os justifique.

## Decisions

1. **Toolchain por `rust-toolchain.toml` com canal exato `1.85`**, não `stable`.
   - *Por quê:* reprodutibilidade absoluta e gate MSRV efetivo — se alguém usar funcionalidade mais nova, o build pinado quebra no próprio ambiente, antes do CI.
   - *Alternativa descartada:* canal `stable` — flutua com releases e mascara violações de MSRV.

2. **Workspace raiz com membros vazios.**
   - *Por quê:* `[workspace]` solitário já valida todo o encadeamento de comandos (`fmt`, `clippy --workspace`, `test --workspace`) e evita criar 19 stubs vazios que virariam conflitos de merge.
   - *Alternativa descartada:* scaffold completo dos sub-crates agora — churn grande, zero comportamento.

3. **CI em workflow único com jobs separados por gate.**
   - Jobs: formatação; análise estática com `-D warnings`; testes do workspace (inclui doctests); exemplos (build + teste); MSRV dedicado ao 1.85 exato.
   - *Por quê:* falhas ficam atribuídas ao gate específico; jobs independentes rodam em paralelo.
   - *Alternativa descartada:* job monolítico — diagnóstico pior sem ganho real.

4. **Clippy estrito desde o dia zero (`-D warnings`) em vez de adoção progressiva.**
   - *Por quê:* endireitar depois é mais caro; hoje o custo é zero (não há código).

5. **Exemplos como cidadãos testados desde já.**
   - `cargo build/test --examples` no CI mesmo sem exemplos existentes — o gate já está ativo quando o primeiro exemplo aparecer, e alimentará diretamente o mdBook da branch `docs/documentations`.

## Risks / Trade-offs

- [Canal exato `1.85` pode ficar atrás de correções do `stable`] → aceito conscientemente: previsibilidade vale mais que patches automáticos; atualização de toolchain é change explícita revisável.
- [CI sem cache de dependências fica lento conforme crates crescem] → mitigável adiante com cache de `~/.cargo` no Actions; não bloqueia esta mudança.
- [Workspace vazio não exercita `--all-targets` sobre código real] → aceito: os gates provam o encadeamento; cobertura real começa com `sciencekit_common`.

## Migration Plan

Mudança aditiva sobre a raiz do repositório; rollback = reverter o merge. Sem consumidores ainda, sem migração.

## Open Questions

Nenhuma.
