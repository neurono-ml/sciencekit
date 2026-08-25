# Proposal: common-core-foundation

## Why

Todos os algoritmos do roadmap dependem do vocabulário central definido no PRD (§3.3 traits fundamentais, §4 memória, §5 concorrência, §9 erros). Uma exploração de design aprofundada decidiu os contratos que atravessam a biblioteca inteira — dtypes genéricos, separação estimador/modelo ajustado, views densas/esparsas/alvos com seam de conversão aberto a terceiros, resolução de modo de execução e batches owned para streaming. Esses contratos precisam existir, testados e estáveis, antes do primeiro estimador (Fase 1); mudá-los depois custaria breaking changes em toda a superfície.

## What Changes

- Criação do crate **`crates/sciencekit_common`** (primeiro sub-crate do workspace), contendo:
  - Trait selada **`SKFloat`** — bound numérico único da biblioteca (dtype genérico por parâmetro de trait).
  - **Erro central `SKError`** (`thiserror`) com taxonomia acordada; enums de erro por algoritmo derivam via `From<SKError>`.
  - Views de fronteira zero-copy: **`SKDataView<'_, F>`** (denso/esparsa, `#[non_exhaustive]`) e **`SKTargetView`** (contínuo/inteiro/nominal), ambas acessíveis via seam `TryInto`, com rejeição precisa em runtime de representação não suportada.
  - Tabela de rótulos (**`SKLabelTable`**) + helpers de canonicalização usados por classificadores (base dos codecs explícitos da Fase 1).
  - Contratos de ajuste: **`SKUnsupervisedFit`** / **`SKSupervisedFit`** (estimador configurado → modelo ajustado como tipo distinto) e **`SKFeatureTransformer`** com tipo associado de saída (requisito antecipado do pipeline type-safe da Fase 6).
  - Contratos de avaliação: **`SKSupervisedScorer`** / **`SKUnsupervisedScorer`** com entrada pura (`score_from_predictions`/equivalente) + método provido conveniente (`score`) que executa inferência.
  - Execução: enum **`SKExecutionMode`** (intenção) ≠ struct **`SKExecutionPlan`** (plano resolvido), contexto injetável e função pura de resolução (`sk_resolve_execution_plan`), com **erro duro** para modo explícito incompatível com o padrão de acesso declarado.
  - Streaming: struct **`SKDataBatch<F>`** owned + traits **`SKLazySource`** (iterador sequencial de batches) e **`SKMappableSource`** (acesso aleatório O(1)).
- Organização em módulos pasta padronizados desde o início (limite de 200 linhas/arquivo).

Fora de escopo: qualquer algoritmo, builders de estimadores (nascem na Fase 1 com o primeiro estimador), codecs completos encoder/decoder (Fase 1), métricas concretas (crates próprios), bindings Python e backends GPU (changes separadas, pós-validação em CPU).

## Capabilities

### New Capabilities

- `scalar-typing`: trait selada `SKFloat` — contrato numérico único usado por views e traits.
- `error-model`: taxonomia central `SKError`, variantes acordadas e propagação uniforme entre algoritmos via `From`.
- `data-view-boundary`: `SKDataView`/`SKTargetView`, seam `TryInto` (compatível com `Into` via blanket da std), zero-copy obrigatório nos impls nativos, evolução não-breaking por novas variantes (`#[non_exhaustive]`), canonicalização de alvos e tabela de rótulos.
- `estimator-contracts`: semântica das duas traits de fit (retorno de tipo modelo distinto, imutável, `Send + Sync` por construção) e do transformador com saída tipada por associated type.
- `scoring-contracts`: scorers supervisionados/não supervisionados com dupla entrada (previsões existentes vs. inferência embutida provida).
- `execution-planning`: separação intenção/plano, resolução determinística por operação com contexto injetável, erro duro em incompatibilidade explícita.
- `streaming-batches`: batches owned com metadados mínimos e as duas traits de fonte (sequencial/memmap-abstracto).

### Modified Capabilities

(nenhuma — não há specs existentes)

## Impact

- **Código:** novo crate `crates/sciencekit_common`; nenhum crate existente é alterado (não há nenhum).
- **Dependências:** `ndarray`, `sprs`, `num-traits`, `thiserror`, `sysinfo` no `sciencekit_common`. Sem `memmap2` (trait de fonte mapeável permanece abstrata — implementação concreta chega com interop) e sem Tokio/rayon nesta change.
- **Downstream:** todas as changes futuras de algoritmos consomem estes contratos; decisões adiadas conscientemente ficam registradas em design.md (representação interna de nominais, amarração de dtype de alvo contínuo ao `F`, multi-alvo 2D, caveat de inferência de tipos com `TryInto`).
- **Critérios de aceite (PRD §8.7/§10.3):** ainda não há modelos treináveis nem exportação nesta change — os critérios completos ativam com o primeiro estimador. O aceite aqui é: todos os contratos compilam sob `Send`/`Sync` onde prometido, testes companion cobrem conversões de views, canonicalização, resolução determinística (contexto simulado), erro duro de incompatibilidade e o contrato de iteração de batches, tudo com dados mock em `ndarray`/`sprs` e TDD.
