# PRD — `sciencekit`

**Product Requirements Document**

Versão: 0.1.0 (rascunho inicial)
Data: 2026-08-24
Status: Aprovada para implementação

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Interface de Usuário](#2-interface-de-usuário)
3. [Arquitetura e Organização de Código](#3-arquitetura-e-organização-de-código)
4. [Gestão de Memória](#4-gestão-de-memória)
5. [Threads e Concorrência](#5-threads-e-concorrência)
6. [Bibliotecas e Dependências](#6-bibliotecas-e-dependências)
7. [Otimização Tecnológica](#7-otimização-tecnológica)
8. [Exportação e Importação de Modelos](#8-exportação-e-importação-de-modelos)
9. [Erros e Observabilidade](#9-erros-e-observabilidade)
10. [Metodologia de Desenvolvimento](#10-metodologia-de-desenvolvimento)
11. [Versionamento e Licença](#11-versionamento-e-licença)
12. [Roadmap de Implementação](#12-roadmap-de-implementação)

---

## 1. Visão Geral

### 1.1. Objetivo

`sciencekit` é uma biblioteca de Machine Learning escrita nativamente em Rust que expõe **todos os algoritmos e utilitários do scikit-learn**, reescritos do zero de forma otimizada. Os diferenciais competitivos são:

- **Performance extrema:** código nativo, SIMD, paralelismo de dados, alocadores customizados.
- **Segurança de memória:** garantias do Rust sem garbage collector.
- **Out-of-core nativo:** todos os algoritmos suportam datasets maiores que a RAM, via streaming ou memory-mapping.
- **Concorrência como cidadã de primeira classe:** decisão automática de modo de execução, configurável pelo usuário.

### 1.2. Posicionamento

Biblioteca equivalente ao scikit-learn em cobertura algorítmica, mas com a performance e segurança do Rust. Público-alvo:

- Engenheiros de ML em Rust que precisam de um toolkit completo.
- Equipes Python que precisam de inferência/treinamento de alta performance via bindings.
- Sistemas de produção onde o pickle do scikit-learn é um risco de segurança.

### 1.3. Escopo

**É:**
- Biblioteca Rust (workspace Cargo com sub-crates).
- API Rust nativa idiomática (builder pattern).
- Bindings Python (PyO3) com zero-copy.
- Cobertura completa da taxonomia do scikit-learn.
- Exportação/importação ONNX, Safetensors e JSON debug.

**Não é (neste momento):**
- CLI para execução de workflows *(anotado como melhoria futura)*.
- Framework de deep learning.
- Serviço/servidor de inferência.

---

## 2. Interface de Usuário

### 2.1. API Rust Nativa (primária)

A interface primária é a API Rust nativa, seguindo **builder pattern** para toda configuração:

```rust
// Exemplo ilustrativo da interface esperada
let modelo = SKKMeansClassifierBuilder::new()
    .number_of_clusters(8)
    .maximum_iterations(300)
    .execution_mode(SKExecutionMode::Automatic)
    .build()?;

modelo.fit(&training_data_view)?;
let predictions = modelo.predict(&test_data_view)?;
```

Princípios:

- **Builder obrigatório** para todos os estimadores, transformadores e pipelines. Construtores diretos ficam privados.
- **Zero-copy nas entradas:** APIs públicas recebem `ArrayView`, `CowArray` ou views esparsas (`sprs`), nunca `Array` por valor.
- **Type-safe pipelines:** validação de compatibilidade entre etapas em tempo de compilação via tipos associados.
- **Nomes completos e explicativos:** sem abreviações em nenhum item público ou privado — única exceção: o prefixo do projeto `sk`/`SK` (ver seção 3.4).

### 2.2. Bindings Python (PyO3)

Interface secundária, porém obrigatória para cada algoritmo:

- **Timing:** logo após um algoritmo estar completo e validado em Rust, uma change associada cria a interface Python correspondente, com os devidos testes. Nenhum algoritmo é considerado "pronto" sem binding Python.
- **API idiomática própria** seguindo padrão builder (não é drop-in replacement do sklearn).
- **Máximo zero-copy:** conversão NumPy ↔ ndarray via views sem cópia sempre que possível (`PyReadonlyArray` → `ArrayView`).
- Implementada no crate `sciencekit_python`.

### 2.3. CLI

Não haverá CLI nesta fase. Registrado no roadmap como melhoria futura (execução de workflows declarativos).

---

## 3. Arquitetura e Organização de Código

### 3.1. Workspace Cargo

Projeto dividido em sub-crates coesos para otimizar tempo de compilação e isolar dependências:

```
sciencekit/
├── Cargo.toml              # workspace root
├── crates/
│   ├── sciencekit/                  # crate umbrella (re-exports)
│   ├── sciencekit_common/           # traits base, tipos, erros centrais
│   ├── sciencekit_math/             # álgebra linear, BLAS, SIMD, esparso, pairwise
│   ├── sciencekit_preprocessing/    # scalers, encoders, polynomial features
│   ├── sciencekit_impute/           # imputadores simples e multivariados
│   ├── sciencekit_linear_model/     # regressões lineares, logística, SGD
│   ├── sciencekit_neighbors/        # SKKNeighborsClassifier/Regressor, SKKDTree, SKBallTree
│   ├── sciencekit_svm/              # SKSVC, SKSVR, SKLinearSVC
│   ├── sciencekit_tree/             # árvores de decisão
│   ├── sciencekit_ensemble/         # random forest, bagging, gradient boosting
│   ├── sciencekit_cluster/          # SKKMeans, SKDBSCAN, SKMiniBatchKMeans
│   ├── sciencekit_decomposition/    # SKPCA, SKTruncatedSVD
│   ├── sciencekit_outlier/          # detecção de anomalias
│   ├── sciencekit_model_selection/  # SKKFold, sk_train_test_split, SKGridSearchCV
│   ├── sciencekit_pipeline/         # SKPipeline type-safe, DAGs
│   ├── sciencekit_metrics/          # métricas de avaliação
│   ├── sciencekit_interop/          # ONNX, Safetensors, traits de I/O
│   ├── sciencekit_gpu/              # SKComputeBackend + OpenCL/CUDA/ROCm
│   └── sciencekit_python/           # bindings PyO3
└── docs/
```

### 3.2. Regras estruturais

| Regra | Detalhe |
|---|---|
| **Uma responsabilidade por módulo** | Módulo pequeno o suficiente para ter exatamente uma responsabilidade. |
| **Limite de 200 linhas** | Arquivo `.rs` que ultrapassa 200 linhas torna-se **módulo pasta**: `meu_modulo/mod.rs` (ou `meu_modulo.rs` + diretório) com sub-módulos de organização fiel e padronizada. |
| **Single responsibility por item** | Funções, structs e traits fazem uma única coisa. Preferir **composição de traits** a herança de responsabilidades. |
| **PRs pequenas** | Cada change/PR altera o mínimo necessário. Evitar PRs com muitas mudanças. |

Organização padrão de módulo pasta:

```
standard_scaler/
├── mod.rs                    # re-exports públicos e docs do módulo
├── builder.rs                # SKStandardScalerBuilder
├── core_implementation.rs    # lógica central do transformador
├── fitting_logic.rs          # fase fit
├── transformation_logic.rs   # fase transform
└── standard_scaler_tests.rs  # testes companion
```

### 3.3. Traits fundamentais (`sciencekit_common`)

Composição de traits como base de toda a biblioteca:

```rust
pub trait SKEstimator { /* hiperparâmetros e fit */ }
pub trait SKPredictor: SKEstimator { /* predict */ }
pub trait SKTransformer: SKEstimator { /* transform / fit_transform */ }

pub trait SKDataSource { /* eager: acesso total em memória */ }
pub trait SKLazySource { /* streaming: Iterator<Item = Batch> */ }
pub trait SKMappableSource { /* memmap: acesso aleatório O(1) */ }
pub trait SKToOnnx { /* exportação ONNX */ }
pub trait SKComputeBackend { /* abstração de dispositivo de cálculo */ }
```

Algoritmos compõem exatamente as traits que suportam. Um `SKSGDClassifier` implementa `SKPredictor` + `SKLazySource`; um `SKStandardScaler` implementa `SKTransformer`.

### 3.4. Convenção de nomenclatura

- **Sem abreviações, com uma única exceção: o prefixo do projeto `sk`/`SK`.** Nomes de módulos, funções, variáveis, structs, traits e qualquer objeto Rust são completos e explicativos no contexto. Exemplos: `maximum_number_of_iterations` (não `max_iter`), `nearest_neighbors_count` (não `k`). O prefixo `sk` é a única abreviação permitida; em itens internos/privados seu uso é opcional, em itens públicos é obrigatório (regra abaixo).
- **Prefixo obrigatório em itens públicos:** todo objeto público (acessível externamente) criado neste projeto recebe o prefixo do projeto:
    - Structs e traits: `SK` + PascalCase — ex.: `SKEstimator`, `SKStandardScaler`, `SKExecutionMode`;
    - Funções públicas de escopo livre (fora de `impl`), variáveis e módulos públicos: `sk_` + snake_case — ex.: `sk_train_test_split`;
    - Métodos **não** recebem prefixo — métodos são as funções dentro de `impl` de structs ou traits (equivalente a métodos OO; daqui em diante chamados de métodos): usam apenas nomes completos — ex.: `fit`, `predict`, `execution_mode`.
- Prefixo de crates: `sciencekit_*` (sempre o nome completo, sem abreviar).
- API oculta com `#[doc(hidden)]` não é considerada pública — pode ser refatorada sem breaking change.

### 3.5. Testes

- Testes vivem em **módulos `_tests.rs` companion** dos arquivos de implementação (ex: `standard_scaler_tests.rs` ao lado de `core_implementation.rs`), não inline nem em diretório `tests/` global.
- Dados mock construídos em `ndarray`/`sprs`.
- **TDD obrigatório:** teste primeiro, falha confirmada, depois implementação.

### 3.6. Toolchain

| Item | Valor |
|---|---|
| MSRV | **Rust 1.85** (stable) |
| Edition | **2024** |

---

## 4. Gestão de Memória

### 4.1. Zero-copy

Obrigatório em toda a superfície pública e nos caminhos quentes internos:

- Entradas das traits recebem `ArrayView`, `CowArray` (e equivalentes esparsos de `sprs`) — nunca cópias de matrizes gigantes.
- Transformações mutáveis usam `.map_inplace()` / `CowArray::into_owned` somente quando inevitável.
- Sempre considerar o **layout de memória** (row-major/column-major, contiguidade) em operações binárias — a eficiência de CPU depende disso.

### 4.2. Higher-order functions (obrigatório)

Performance depende de evitar iterações manuais por índice. É obrigatório priorizar:

- `.map()`, `.map_inplace()`, `.zip_mut_with()` para travessias simples;
- `azip!()` e `par_azip!()` para iterações lock-efficient;
- `rayon` para paralelismo de dados.

### 4.3. Alocadores customizados

| Alocador | Feature flag | Plataformas | Características |
|---|---|---|---|
| glibc malloc | (nenhuma — fallback) | todas | Padrão do sistema, zero dependência |
| jemalloc | `allocator-jemalloc` | todas | Arenas por thread; brilha em muitas alocações de mesmo tamanho |
| mimalloc | `allocator-mimalloc` | todas | Latência p99 muito baixa; tamanhos variáveis |

- Ambos disponíveis em **todas as plataformas**; usuário escolhe no builder.
- O **mecanismo de decisão automática** (seção 5.3) recomenda/seleciona o alocador por padrão conforme plataforma e workload detectado.
- Ganhos esperados vs glibc em workloads ML: 15–30% (throughput de alocação).

### 4.4. Out-of-core: duas traits complementares

Todos os algoritmos suportam out-of-core — é diferencial da biblioteca. Cada algoritmo **declara seu padrão de acesso** e implementa a(s) trait(s) que funcionam para ele:

| Trait | Estratégia | Algoritmos típicos |
|---|---|---|
| `SKLazySource` | Streaming sequencial em batches (`Iterator<Item = Batch>`) | SKSGD*, SKMiniBatchKMeans, SKPCA incremental |
| `SKMappableSource` | Memory-mapped files (`memmap2`), acesso aleatório O(1) | SKKMeans, SKKNN, SKDBSCAN, SKDecisionTree |

- Se o algoritmo puder usar ambas, implementa ambas; senão, implementa apenas a que funciona.
- Formato mapeável: binário contíguo com layout conhecido (compatível com o formato interno de modelos, seção 8).

### 4.5. Arrays esparsos

Suporte desde o início via **`sprs` + `ndarray`** (CSR/CSC/COO). Essencial para Lasso, SVMs lineares e classificação de texto. As traits centrais aceitam views densas ou esparsas conforme o algoritmo.

---

## 5. Threads e Concorrência

### 5.1. Separação CPU / I/O

Regra inegociável: **nunca bloquear threads assíncronas com trabalho intensivo de CPU.**

- Processamento matemático roda em pool dedicado de CPU (**rayon**).
- I/O assíncrono (rede, streams) usa runtime próprio (Tokio), delegando blocos de computação ao pool de CPU.

### 5.2. Paralelismo de dados

- Interno dos algoritmos (multiplicações, distâncias, agregações): `rayon` + `par_azip!`.
- Habilitado/desabilitado via **feature flags** (por capacidade).
- Pool de threads: default do rayon; **configurável** pelo usuário (número de threads, pinning) via builder.

### 5.3. Mecanismo de decisão automática de execução

Todo builder expõe `execution_mode(SKExecutionMode::...)`. Default: `Automatic`.

**Modos de execução:**

| Modo | Descrição |
|---|---|
| `InProcessSynchronous` | Dataset cabe na RAM; algoritmo eager; rayon only |
| `InProcessAsynchronous` | Fonte de I/O assíncrono (rede/stream); Tokio orquestra, rayon calcula |
| `OutOfCoreStreaming` | Dataset > RAM; batches sequenciais de disco (`SKLazySource`) |
| `OutOfCoreMemoryMapped` | Dataset > RAM; acesso aleatório necessário (`SKMappableSource`) |

**Parâmetros usados pela decisão automática:**

- `available_memory`: RAM livre (detectada via sysinfo ou informada);
- `dataset_size`: amostras × features × bytes por elemento;
- `algorithm_access_pattern`: declarado pelo algoritmo (sequential/random/iterative);
- `cpu_cores`: via `std::thread::available_parallelism()`;
- `batch_size_hint`: opcional, informado pelo usuário.

**Comportamento:** defaults automáticos e transparentes; o usuário pode sobrescrever qualquer parâmetro no builder (configuração do `SKPipeline` tem menor precedência que a do estimador específico).

---

## 6. Bibliotecas e Dependências

### 6.1. Núcleo

| Crate | Uso |
|---|---|
| `ndarray` | Matrizes densas, views zero-copy, `azip!`/`par_azip!` |
| `sprs` | Arrays esparsos CSR/CSC/COO |
| `rayon` | Paralelismo de dados |
| `tokio` | Runtime assíncrono para I/O |
| `serde` | Base de serialização (análoga à trait de exportação padrão) |
| `thiserror` | Erros customizados por algoritmo |
| `anyhow` | Facilitação em aplicações consumidoras |
| `tracing` | Observabilidade estruturada |
| `memmap2` | Memory-mapped files |
| `sysinfo` | Detecção de RAM disponível |

### 6.2. Álgebra linear (BLAS/LAPACK) — híbrido

- **Padrão:** implementações puras em Rust (`sciencekit_math`), zero dependências C/Fortran.
- **Feature flag `blas-backend`:** troca operações críticas (matmul, SVD, QR) por BLAS/LAPACK.

| Backend | Licença | Política |
|---|---|---|
| OpenBLAS | BSD-3-Clause | Default do flag `blas-backend` |
| BLIS | BSD-3-Clause | Alternativa |
| Intel MKL | Proprietária | **Nunca dependência direta**; opt-in explícito via `blas-mkl`, instalação por conta do usuário |
| Apple Accelerate | System framework | Flag específica macOS, sem redistribuição de binário |

A configuração padrão da biblioteca (sem flags) é pure Rust, livre de dependências proprietárias.

### 6.3. SIMD — híbrido

- **Base:** código `ndarray`/`azip!` escrito para autovectorização do LLVM (cobre ~80% dos casos, estável).
- **Hot paths críticos** (identificados por profiling): crate `wide` (SIMD explícito portável, stable).
- **Futuro:** migrar `wide` → `std::simd` quando estabilizar.

### 6.4. GPU

Abstração via **trait própria `SKComputeBackend`** (em `sciencekit_gpu`) + crates de binding existentes para FFI:

| Backend | Binding | Fase |
|---|---|---|
| OpenCL | `ocl` | Início |
| CUDA | `cudarc` | Início |
| ROCm/HIP | bindings HIP | Início |
| SYCL/oneAPI, Metal, Vulkan | a definir | Futuro |

- CPU é o backend default e sempre presente.
- Suporte a cada backend entra como **change separada, após o algoritmo estar pronto** em CPU.

### 6.5. I/O de dados

- Código depende apenas de **traits arbitrárias de I/O** (fontes/receptores de dados definidas em `sciencekit_interop`).
- Integração via conversões `Into<>`:
  - `Polars DataFrame/LazyFrame` → fonte;
  - `DataFusion DataFrame` → fonte;
  - Arrow/Parquet suportados através desses motores.

### 6.6. Interoperabilidade

| Crate | Uso |
|---|---|
| `pyo3` | Bindings Python (`sciencekit_python`) |
| `safetensors` | Formato interno/exportação de modelos |
| `onnx`/`ort` (a validar na spec técnica) | Exportação/importação ONNX |

---

## 7. Otimização Tecnológica

### 7.1. Camadas de otimização

1. **Algorítmica:** escolher complexidade correta antes de micro-otimizar.
2. **Layout de memória:** contiguidade, SoA vs AoS quando relevante, cache-friendliness.
3. **Higher-order functions:** `azip!`/`par_azip!`/`zip_mut_with` (obligatório — ver 4.2).
4. **SIMD:** autovectorização → `wide` em hot paths profileados.
5. **BLAS:** flag opt-in para álgebra densa pesada.
6. **GPU:** backend adicional por change separada.
7. **Alocadores:** escolha automática/configurável.

### 7.2. Evolução iterativa obrigatória por algoritmo

1. Implementação **naive** (simples, sequencial).
2. Testes de unidade (TDD).
3. Refatoração para **performance** (SIMD, rayon, layout).
4. Refatoração para **streaming/out-of-core** (SKLazySource/SKMappableSource).

Nenhuma etapa pode ser pulada.

### 7.3. Feature flags

Granularidade dupla:

- **Por capacidade:** `parallel`, `allocator-jemalloc`, `allocator-mimalloc`, `blas-backend`, `blas-mkl`, `gpu-opencl`, `gpu-cuda`, `gpu-rocm`, `telemetry-opentelemetry`, ...
- **Por grupo de algoritmo:** `classification`, `regression`, `clustering`, `decomposition`, `preprocessing`, ...

---

## 8. Exportação e Importação de Modelos

### 8.1. Trait de exportação padrão

Análoga ao papel do `serde`: uma trait central de serialização/desserialização de modelos, da qual derivam as implementações concretas (Safetensors interno, JSON debug, ONNX, Safetensors público).

### 8.2. Formato interno: Safetensors estendido

- **Formato único de modelo:** safetensors — header JSON + tensores contíguos com acesso aleatório por nome.
- Header JSON carrega metadados arbitrários:
  - `"sciencekit_format_version"`: versão do formato (para migrações futuras);
  - tipo do algoritmo, hiperparâmetros do builder, estado de treinamento;
  - **metadados de recuperabilidade:** indicam como o modelo pode ser recarregado para continuar treinamento (checkpointing).
- Delimitação clara de onde um modelo inicia/finaliza e de suas partes/operadores.

### 8.3. Escrita parcial (sem rewrite total)

| Caso | Estratégia |
|---|---|
| Atualizar tensor existente (mesmo tamanho) | Escrita in-place no offset conhecido (sempre funciona) |
| Adicionar tensor — modelo grande | **Sharding:** shards `.safetensors` imutáveis + índice externo; novos tensores vão a novo shard |
| Adicionar tensor — modelo pequeno | **Padding de header:** espaço reservado permite append + header in-place enquanto houver padding |

A escolha sharding/padding fica a cargo do mecanismo de decisão automática (tamanho do modelo), configurável.

### 8.4. Compressão

Variantes compactadas do formato interno: `.safetensors.gz`, `.safetensors.brotli`, `.snappy.safetensors`.

### 8.5. ONNX

- **Exportação:** todo estimador/pipeline implementa `SKToOnnx`.
- **Importação (imediato):** modelos ONNX e Safetensors treinados em outras frameworks carregam como um **SKPredictor genérico**, utilizável isoladamente ou dentro de um SKPipeline.
- **Importação (futuro — anotado):** converter modelos ONNX/Safetensors importados para tipos nativos da biblioteca, permitindo retomar treinamento ou criar LoRAs.

### 8.6. JSON debug

Serialização legível para inspeção/debug, seguindo a mesma trait central de exportação.

### 8.7. Validação obrigatória

Toda implementação valida, no mínimo:

1. Execução completa do algoritmo com **muitos dados** e com **poucos dados**;
2. Execução sob **concorrência**;
3. **Exportação** do modelo e **geração de métricas**.

---

## 9. Erros e Observabilidade

### 9.1. Erros

- `thiserror` para enums de erro **por algoritmo** (erros específicos preservados).
- `From<ErroCentral>` implementado nos enums de algoritmo quando fizer sentido — erros comuns (shape mismatch, dtype inválido, E/S) propagam-se uniformemente entre algoritmos.
- `anyhow` recomendado para aplicações consumidoras (ergonomia), não dentro da biblioteca.

### 9.2. Observabilidade

- **`tracing`** como base; `tracing-subscriber` quando necessário.
- Spans/logs orientados ao **formato OpenTelemetry**.
- Suporte out-of-the-box a **OpenTelemetry** via `tracing-opentelemetry`.
- Telemetria **desabilitável** (feature flag + configuração no builder); custo próximo de zero quando desativada.

---

## 10. Metodologia de Desenvolvimento

### 10.1. Fluxo de trabalho

- **OpenSpec** orienta especificação de cada change; **graphify** auxilia navegação/compreensão do grafo de conhecimento do projeto.
- Todo desenvolvimento ocorre em **git worktrees**. **Nenhum commit direto na branch principal.**
- Changes pequenas; preferência explícita por PRs com poucas mudanças.

### 10.2. TDD obrigatório

1. Escrever teste (dados mock em `ndarray`/`sprs`, módulo `_tests.rs` companion).
2. Confirmar falha.
3. Implementar o mínimo para passar.
4. Refatorar (performance → streaming), mantendo testes verdes.

### 10.3. Checklist de aceite de qualquer implementação

- [ ] Executa corretamente com muitos dados e com poucos dados;
- [ ] Correto sob concorrência (modo automático + modos explícitos aplicáveis);
- [ ] Exporta modelo (Safetensors mínimo) e produz métricas;
- [ ] Testes `_tests.rs` companion cobrindo tudo acima;
- [ ] Nomes completos, sem abreviações (única exceção: prefixo sk/SK — §3.4);
- [ ] Itens públicos com prefixo obrigatório SK/sk_ conforme §3.4 (métodos dentro de `impl` não recebem prefixo);
- [ ] Nenhum arquivo > 200 linhas sem virar módulo pasta padronizado;
- [ ] Binding Python criado em change associada (quando aplicável);
- [ ] Backend(s) GPU adicionado(s) em change separada posterior (quando aplicável).

---

## 11. Versionamento e Licença

### 11.1. Versionamento

- **SemVer** com prefixo `0.x` durante desenvolvimento/instabilidade.
- Migração para SemVer estável (1.x) quando a API atingir maturidade.
- Breaking changes em API pública só entre minor/major conforme SemVer; itens `#[doc(hidden)]` podem mudar livremente.

### 11.2. Formato de modelo

- `"sciencekit_format_version"` no header JSON de cada arquivo de modelo; loaders suportam leitura de versões anteriores.

### 11.3. Licença

- **Apache-2.0** (proteção contra patentes incluída).

---

## 12. Roadmap de Implementação

Ordem fiel ao mapeamento do scikit-learn (do handoff), respeitando dependências entre crates:

### Fase 0 — Fundações
1. Workspace, toolchain (MSRV 1.85, edition 2024), CI, licença.
2. `sciencekit_common`: traits `SKEstimator`, `SKPredictor`, `SKTransformer`, `SKDataSource`, `SKLazySource`, `SKMappableSource`; erros centrais; tipos.
3. `sciencekit_math`: higher-order ops, layouts, pairwise distances, esparso (`sprs`), interface SIMD/BLAS.
4. Mecanismo de decisão automática de execução + builders base + tracing/OTel.

### Fase 1 — Pré-processamento
5. `sciencekit_preprocessing`: `SKStandardScaler`, `SKMinMaxScaler`, `SKRobustScaler`, `SKOneHotEncoder`, `SKPolynomialFeatures`.
6. `sciencekit_impute`: estratégias simples, `SKKNNImputer`.

### Fase 2 — Modelos lineares
7. `sciencekit_linear_model`: `SKLinearRegression`, `SKRidge`, `SKLasso`, `SKElasticNet`, `SKLogisticRegression`; `SKSGDClassifier`/`SKSGDRegressor` com `SKLazySource`.

### Fase 3 — Vizinhança e SVM
8. `sciencekit_metrics` (pairwise já em math) + `sciencekit_neighbors`: SKKNeighborsClassifier/SKKNeighborsRegressor, SKKDTree, SKBallTree.
9. `sciencekit_svm`: SKSVC, SKSVR, SKLinearSVC.

### Fase 4 — Árvores e ensembles
10. `sciencekit_tree`: SKDecisionTreeClassifier/SKDecisionTreeRegressor.
11. `sciencekit_ensemble`: agregação paralela com rayon (SKRandomForest, SKBagging, SKGradientBoosting).

### Fase 5 — Não supervisionado
12. `sciencekit_cluster`: SKKMeans, SKDBSCAN, SKMiniBatchKMeans (streaming).
13. `sciencekit_decomposition`: SKPCA, SKTruncatedSVD.
14. `sciencekit_outlier`: detecção de anomalias.

### Fase 6 — Seleção, pipeline, métricas
15. `sciencekit_model_selection`: SKKFold, sk_train_test_split, SKGridSearchCV.
16. `sciencekit_pipeline`: SKPipeline type-safe (tipos associados), DAGs.
17. `sciencekit_metrics` completa: accuracy, f1, MSE, matriz de confusão.

### Fase 7 — Interop e produção
18. `sciencekit_interop`: safetensors (interno, sharding/padding, compressão), ONNX export/import, JSON debug, traits de I/O + Polars/DataFusion.
19. `sciencekit` umbrella: re-exports e documentação unificada.
20. `sciencekit_python`: bindings PyO3 completos.

### Transversal (contínua)

- **Por algoritmo concluído:** binding Python em change associada; backend GPU (OpenCL → CUDA → ROCm) em changes separadas seguintes.
- **Melhorias futuras registradas (fora de escopo agora):** CLI de workflows; importação ONNX/Safetensors convertida para tipos nativos com retreinamento/LoRA.

---

*Documento gerado a partir de decisões consolidadas com o proprietário do produto. Alterações relevantes de escopo exigem atualização desta PRD antes da spec técnica.*
