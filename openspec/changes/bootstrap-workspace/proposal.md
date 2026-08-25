# Proposal: bootstrap-workspace

## Why

O repositório está recém-inicializado e não possui workspace Cargo, toolchain fixada nem CI — pré-requisito estrutural do item 1 da Fase 0 do PRD (§12). Toda mudança subsequente (começando por `sciencekit_common`) precisa de um ambiente de build reprodutível (Rust 1.85 / edition 2024), de gates de qualidade automatizados e da licença Apache-2.0 definidos antes de existir qualquer código.

## What Changes

- Criação do **workspace Cargo raiz** (`Cargo.toml` com `[workspace]`, membros vazios inicialmente — cada sub-crate nasce na sua própria change, mantendo PRs pequenas).
- Fixação de toolchain via `rust-toolchain.toml`: canal exato **1.85**, profile mínimo, alvo host — garantindo edition 2024 e MSRV reproduzíveis.
- Adição da **licença Apache-2.0** (`LICENSE`).
- Criação do **workflow de CI** (GitHub Actions) com os gates acordados:
  - `cargo fmt --check`;
  - `cargo clippy --workspace --all-targets -- -D warnings`;
  - `cargo test --workspace` (inclui módulos companion `*_tests.rs` e doctests);
  - build + teste de exemplos (`cargo build --examples && cargo test --examples`);
  - **gate MSRV**: job separado compilando com o toolchain 1.85 exato.
- Ajustes pontuais em `.gitignore` para artefatos de build Rust.

Fora de escopo: criação de sub-crates, configuração de proteção de branch no GitHub (decisão administrativa do repositório remoto), perfis de release e ferramentas auxiliares (`cargo-deny`, cobertura).

## Capabilities

### New Capabilities

- `workspace-bootstrap`: comportamentos verificáveis da fundação do repositório — toolchain pinada e reprodutível, gates de CI obrigatórios para qualquer PR, licença Apache-2.0 presente e declarada nos manifests.

### Modified Capabilities

(nenhuma — não há specs existentes)

## Impact

- **Arquivos:** raiz do repositório (`Cargo.toml`, `rust-toolchain.toml`, `LICENSE`, `.gitignore`) e `.github/workflows/ci.yml`.
- **Dependências:** nenhuma dependência de código nesta change (workspace vazio).
- **Sistemas:** GitHub Actions passa a executar em pull requests; todos os changes futuros assumem estes gates como pré-condição.
- **Critérios de aceite (PRD §8.7/§10.3):** os critérios algorítmicos (muitos/poucos dados, concorrência, exportação + métricas) ainda não se aplicam pois não existem algoritmos; aplicam-se a partir da primeira change de estimador (Fase 1). O aceite desta change é: clone limpo compila, testa e passa em todos os gates no toolchain 1.85 exato, com CI executando-os automaticamente em PR.
