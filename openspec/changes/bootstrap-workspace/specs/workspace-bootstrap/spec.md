## Purpose

Estabelece a fundação verificável do repositório sciencekit: toolchain Rust fixada e reprodutível, portão de qualidade automatizado em CI para qualquer pull request e licença Apache-2.0 declarada. Toda mudança futura assume estes comportamentos como pré-condição.

## ADDED Requirements

### Requirement: Toolchain fixada e reprodutível
O repositório SHALL fixar a versão exata do toolchain Rust (1.85) de forma declarativa, de modo que qualquer clone limpo compile com a mesma versão, usando edition 2024, sem dependência do Rust instalado na máquina do contribuidor.

#### Scenario: Clone limpo usa o toolchain correto
- **WHEN** um clone limpo do repositório executa qualquer comando Cargo sem configuração adicional
- **THEN** o toolchain 1.85 exato é selecionado automaticamente a partir da declaração versionada no repositório

#### Scenario: Código incompatível com edition 2024 é rejeitado no build
- **WHEN** código-fonte que exige edição anterior à 2024 é compilado no ambiente do projeto
- **THEN** a compilação falha, evidenciando que edition 2024 está em vigor

### Requirement: Portão de qualidade em CI para todo pull request
O CI SHALL executar, automaticamente em cada pull request, verificações de formatação, análise estática com avisos tratados como erros, suíte completa de testes do workspace (incluindo testes de documentação) e build/teste dos exemplos. Qualquer falha SHALL impedir a aprovação automática da mudança.

#### Scenario: Formatação divergente reprova o PR
- **WHEN** um pull request contém código fora do padrão de formatação definido
- **THEN** o job de verificação de formatação falha e aponta as divergências

#### Scenario: Aviso de análise estática reprova o PR
- **WHEN** o código introduz qualquer aviso de análise estática
- **THEN** o job de análise falha, pois avisos são promovidos a erros

#### Scenario: Falha de teste reprova o PR
- **WHEN** qualquer teste do workspace ou de exemplos falha no CI
- **THEN** o pull request não pode ser considerado validado pela automação

### Requirement: Compatibilidade garantida com o MSRV
O CI SHALL incluir uma verificação dedicada que compila e testa o projeto usando exclusivamente a versão mínima suportada (1.85), de modo que uso acidental de funcionalidades de Rust mais recentes seja detectado antes do merge.

#### Scenario: Funcionalidade mais nova que o MSRV é detectada
- **WHEN** código utiliza funcionalidade estável introduzida após 1.85
- **THEN** o job de MSRV falha enquanto os demais jobs com toolchain pinado também evidenciam a incompatibilidade

### Requirement: Licença Apache-2.0 presente e declarada
O repositório SHALL conter o texto completo da licença Apache-2.0 e os manifests do workspace SHALL declará-la como licença do projeto.

#### Scenario: Licença visível no repositório
- **WHEN** o repositório é inspecionado
- **THEN** o arquivo de licença Apache-2.0 existe na raiz e a declaração de licença consta nos manifests do workspace
