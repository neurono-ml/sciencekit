# Tasks — bootstrap-workspace

## 1. Fundação do workspace

- [ ] 1.1 Criar `Cargo.toml` raiz com `[workspace]` (membros vazios), `resolver` apropriado à edição e `[workspace.package]` declarando licença Apache-2.0
- [ ] 1.2 Criar `rust-toolchain.toml` fixando canal exato `1.85`
- [ ] 1.3 Adicionar `LICENSE` com o texto completo Apache-2.0
- [ ] 1.4 Atualizar `.gitignore` com artefatos de build Rust (`target/`, entre outros)

## 2. Workflow de CI

- [ ] 2.1 Criar workflow de CI disparado em pull request, com job de formatação (`cargo fmt --check`)
- [ ] 2.2 Adicionar job de análise estática com avisos promovidos a erro (`cargo clippy --workspace --all-targets -- -D warnings`)
- [ ] 2.3 Adicionar job de testes do workspace incluindo testes de documentação
- [ ] 2.4 Adicionar job de build e teste de exemplos
- [ ] 2.5 Adicionar job dedicado de MSRV compilando com toolchain 1.85 exato

## 3. Validação de aceite

- [ ] 3.1 Em clone limpo do worktree, executar todos os comandos dos gates localmente e confirmar verde ponta a ponta (build, fmt, clippy, testes, exemplos)
- [ ] 3.2 Confirmar que edition 2024 está em vigor e que o toolchain selecionado automaticamente é o 1.85 fixado
- [ ] 3.3 Abrir PR e verificar que todos os jobs de CI executam e passam no pull request
