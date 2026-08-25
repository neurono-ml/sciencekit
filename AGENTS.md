# AGENTS.md

Instruções para agentes trabalhando neste repositório. Fonte de verdade de produto: `docs/PRD.md`.

## Visão geral

- `sciencekit`: biblioteca de ML em Rust reimplementando todo o scikit-learn, com performance extrema, zero-copy e out-of-core nativo.
- Repo recém-inicializado: ainda não existe workspace Cargo. A Fase 0 do roadmap (PRD §12) cria o workspace com sub-crates `sciencekit_*` em `crates/`.
- Toolchain alvo: Rust 1.85, edition 2024. O PRD prevalece sobre `docs/handoff.md`, que contém decisões obsoletas (ex.: MSRV 1.64).
- `graphify-out/` contém o grafo de conhecimento do codebase — questões sobre arquitetura/relações entre arquivos devem passar pela skill graphify (`/graphify`) antes de explorar manualmente.

## Fluxo de trabalho (obrigatório)

- **Nenhum commit direto na `main`.** Toda change é implementada em git worktrees:
  - Código: worktree `temporary/worktrees/<type>/<change-name>`, branch `<type>/<change-name>` (`type` ∈ `feat|bugfix|chore|docs`).
  - Definições OpenSpec: worktree `temporary/worktrees/<type>/<change-name>-openspec`, branch `<type>/<change-name>-openspec`. Os arquivos de `openspec/` da change são commitados nessa branch associada.
  - `temporary/` está no `.gitignore` — worktrees não são versionadas.
- **TDD obrigatório:** toda task usa a skill `tdd` — teste primeiro, confirmar falha, implementação mínima, refatoração.
- **Revisão independente:** ao final de cada change, um agente independente revisa o resultado validando se a spec foi efetivamente atendida, antes do PR.
- **Pós-merge:** quando um PR é mergeado, o agente opencode no GitHub mergeia a branch `-openspec` correspondente em `main` e executa sync + archive da change.

## Planejamento (OpenSpec)

- Usar os comandos `/opsx-propose`, `/opsx-apply`, `/opsx-archive` (`.opencode/commands/`); CLI `openspec` instalada.
- Criar changes em grupos separados:
  1. implementação do algoritmo;
  2. changes para os diferentes acelerators (GPU OpenCL/CUDA/ROCm, BLAS/SIMD, alocadores, bindings Python);
  3. changes para documentação.
- Binding Python e backend GPU entram como changes associadas/separadas, somente depois do algoritmo pronto e validado em CPU.

## Convenções de código (do PRD)

- Builder pattern obrigatório; construtores diretos privados. Todo builder expõe `execution_mode(SKExecutionMode::...)` com default `Automatic`.
- **Sem abreviações** em qualquer nome Rust, com uma única exceção: o prefixo do projeto `sk`/`SK`. Exemplos: `maximum_number_of_iterations`, não `max_iter`; `nearest_neighbors_count`, não `k`.
- **Prefixo obrigatório em itens públicos** (regra completa no PRD §3.4): structs e traits usam `SK` + PascalCase (`SKEstimator`, `SKStandardScaler`); funções públicas de escopo livre (fora de `impl`), variáveis e módulos públicos usam `sk_` + snake_case (`sk_train_test_split`). Métodos — funções dentro de `impl` de structs ou traits — não recebem prefixo. Crates mantêm sempre o nome completo (`sciencekit`, `sciencekit_*`).
- Zero-copy nas APIs públicas: `ArrayView`/`CowArray`/views esparsas (`sprs`), nunca `Array` por valor.
- Arquivo `.rs` > 200 linhas vira módulo pasta padronizado (`mod.rs`, `builder.rs`, `core_implementation.rs`, `fitting_logic.rs`, `*_tests.rs`).
- Testes em módulos companion `*_tests.rs` ao lado da implementação; dados mock em `ndarray`/`sprs`. Nunca inline nem diretório `tests/` global.
- Evolução iterativa por algoritmo, sem pular etapas: naive → testes → performance (SIMD/rayon/layout) → streaming/out-of-core.
- Aceite de toda implementação (PRD §8.7): roda com muitos e poucos dados, sob concorrência, exporta modelo e produz métricas.
- CPU nunca bloqueia threads async (rayon para cálculo, Tokio para I/O); iterações via `.map()`/`azip!()`/`par_azip!()`, nunca loops manuais por índice.

## Documentação

- Branch principal de documentação: `docs/documentations`. Worktrees de docs são mergeadas nela (não em `main`).
- Essa branch hospeda um mdBook compatível com GitHub Pages contendo: API, exemplos de uso e descrição de cada função.
- Testes de unidade das funções da API e testes e2e são usados como exemplos no book.

## Skill de agente

- Branch `chore/skill`: skill seguindo https://agentskills.io/specification, compatível com a versão da biblioteca em `main`, exposta conforme capacidades novas são desenvolvidas.
- Estruturada em arquivos separados por capacidade, com scripts e assets de apoio, para que agentes IA simples ou avançados consigam usar a biblioteca.

## Releases

- Cada versão da biblioteca recebe tag Git, é publicada no crates.io e associada a uma GitHub Release.
