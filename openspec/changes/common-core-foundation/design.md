# Design — common-core-foundation

## Context

O crate `sciencekit_common` nasce sobre o workspace da change `bootstrap-workspace`. Os contratos aqui destilados foram decididos em exploração de design prévia e estão formalizados nos specs deste diretório (ver proposal — Capabilities). Restrições estruturais do PRD que moldam o desenho: zero-copy na superfície pública (§4.1), composição de traits (§3.3), nomenclatura completa com prefixo `sk`/`SK` (§3.4), arquivos de até 200 linhas virando módulo pasta padronizado (§3.2), testes companion `*_tests.rs` (§3.5).

## Goals / Non-Goals

**Goals:**
- Congelar o vocabulário de tipos e traits que todos os crates futuros compilam contra.
- Garantir por construção (`Send`/`Sync`, tipos distintos estimador/modelo) as propriedades de concorrência prometidas.
- Manter a fronteira aberta a integradores externos (seam único de conversão falível).
- Resolução de execução testável sem máquina real (contexto injetável).

**Non-Goals:**
- Qualquer algoritmo ou builder de estimador (Fase 1+).
- Codecs completos encoder/decoder (Fase 1 — aqui só a tabela de rótulos e a canonicalização).
- Implementação concreta de memmap (interop), paralelismo (rayon) e I/O assíncrono (tokio).

## Decisions

1. **Dtype genérico como parâmetro de trait, não tipo associado; trait selada `SKFloat` como único bound numérico.**
   - *Por quê:* cada algoritmo terá kernels distintos por dtype (SIMD `wide` usa tipos concretos); múltiplas impls por dtype são naturais, e um único bound selado impede fragmentação de bounds.
   - *Alternativas:* tipo associado (impede múltiplos dtypes por tipo concreto sem maquinaria extra); `f64` concreto (mais simples, mas fecha a porta ao f32 exigido por performance/GPU futuros).

2. **Duas traits de ajuste separadas por supervisão; ajuste retorna modelo distinto; ajuste opera sobre referência compartilhada.**
   - *Por quê:* imutabilidade pós-ajuste dá `Send`/`Sync` por construção, validação cruzada e busca em grade paralelizam sem lock, prever-sem-ajustar não compila, e pipelines type-safe (PRD §2.1) exigem tipos encadeáveis estaticamente. Referência compartilhada no ajuste permite reuso do mesmo estimador em múltiplos folds sem clonar.
   - *Alternativas:* objeto mutável estilo sklearn (`&mut self`) — pior para concorrência e erros de uso em runtime; trait única genérica nos alvos com `()` — uniforme, mas o call site `fit(x, ())` foi rejeitado pelo proprietário do produto.
   - *Convenção de nomes:* estimador configurado `SKXxx`; modelo treinado `SKXxxModel` — alinhado à exportação (quem exporta Safetensors é o modelo; hiperparâmetros vão ao header JSON, PRD §8.2). O exemplo do PRD §2.1 será atualizado quando esta convenção for formalizada em release.

3. **Fronteira de dados: enums wrapper `#[non_exhaustive]` com despacho único na entrada; seam universal via bound falível da std (`TryInto`).**
   - *Por quê:* o seam dá aos terceiros UM ponto de integração (implementar conversão para nossas views) válido tanto para conversões infalíveis (promovidas automaticamente pela std) quanto falíveis; o erro flui para o `Result` já retornado pelas operações. Despacho denso/esparsa uma vez por operação mantém kernels limpos; algoritmos parcialmente suportantes rejeitam variante com erro específico antes de processar elementos.
   - *Alternativas:* traits com entrada fixa por representação (type-safety marginal, mata o ponto de extensão); bound infalível apenas (sem porta para validação de terceiros).
   - *Contrato zero-copy:* implementações nativas de conversão nunca copiam dados de matriz; cópias existem apenas como métodos explícitos de materialização.
   - *Intermediários de pipeline:* conversão a partir de blocos owned empresta os dados dentro do escopo da chamada — requisito antecipado do encadeamento estático da Fase 6.

4. **Alvos: armazenamento ≠ interpretação.**
   - *Por quê:* `[1,2,3]` é contínuo para o regressor e categórico para o classificador; a view descreve como o dado está guardado (contínuo/inteiro/nominal), o algoritmo decide o significado. Elevação inteiro→contínuo lossless é permitida; aritmética numérica sobre rótulos canônicos nunca acontece (canonicalização produz índices simbólicos).
   - *Canonicalização:* função livre determinística → índices compactos + tabela reversível; é a base da codificação automática de classificadores (Fase 1+) e dos codecs explícitos cujo treino será atômico com metades derivadas (encoder/decoder compartilhando tabela imutável).
   - *Dtype dos alvos contínuos desacoplado do dtype das features:* a view contínua guarda `f64` independentemente do parâmetro `F` das features; regressores convertem uma única vez na entrada. Evita contaminar toda a maquinaria de alvos com o parâmetro genérico. Alternativa (amarrar a `F`) reaberta se benchmarks mostrarem custo real.

5. **Scorers com dupla entrada e método provido; moram no vocabulário central.**
   - *Por quê:* forma pura (alvos verdadeiros × previsões existentes) evita re-inferência; forma conveniente (modelo × features × alvos) alimenta GridSearchCV/pipelines; método provido delega à pura após inferir, então autores de scorer implementam só a métrica. Ambas retornam resultado falível (inferência pode falhar). Definidos no crate comum desde já porque são parte do contrato público estável; implementações concretas chegam com as métricas.
   - *Fronteira registrada:* funções de perda de otimização (boosting, Fase 4) são maquinário interno de ensembles — NÃO usam os scorers.

6. **Execução: intenção (`SKExecutionMode`) ≠ plano (`SKExecutionPlan`); resolução pura por operação; erro duro em incompatibilidade explícita.**
   - *Por quê:* o tamanho do conjunto só é conhecível na entrada da operação — ajuste e predição resolvem planos independentes semeados pela mesma intenção guardada no estimador/modelo. Resolução como função pura sobre contexto injetável torna o comportamento testável e determinístico. Falha dura preserva a semântica de pedidos explícitos; automático sempre escolhe modo compatível e nunca erra por incompatibilidade.
   - *Contexto:* memória disponível, núcleos, tamanho do conjunto, padrão de acesso declarado pelo algoritmo, dica de lote. Leitura física da máquina fica confinada ao construtor padrão do contexto (`sysinfo`); a lógica de decisão jamais lê ambiente.

7. **Streaming: batches owned com metadados mínimos; fonte sequencial iterável e falível; fonte aleatória abstrata sem dependência de mapeamento.**
   - *Por quê:* bloco owned atravessa fronteiras de thread (pipeline I/O ∥ CPU do PRD §5.1); empréstimo amarraria o processamento ao leitor. O contrato aleatório fica independente de `memmap2` — implementações mapeadas pertencem à interop.

8. **Organização modular desde o primeiro commit:** pastas por conceito (`sk_float/`, `errors/`, `data_view/`, `target_view/`, `label_table/`, `fit_traits/`, `scorer_traits/`, `execution/`, `batching/`), `mod.rs` com re-exports públicos, testes companion ao lado, nenhum arquivo além de 200 linhas.

9. **Dependências mínimas:** `ndarray`, `sprs`, `num-traits`, `thiserror`, `sysinfo`. Sem tokio/rayon/memmap2/serde nesta change.

## Risks / Trade-offs

- [Bound falível pode degradar inferência de tipos em alguns call sites] → mitigado: casos nativos têm impls diretas; se atrito real surgir, adiciona-se açúcar construtor próprio sem mudança de contrato.
- [Monomorfização por dtype multiplica código gerado] → aceito: só dtypes instanciados compilam; revisão de tempo de build entra como critério nas changes de algoritmos.
- [`sysinfo` pesa no grafo de dependências do crate comum] → mitigado: leitura física isolada no construtor padrão de contexto; caminhos de teste injetam valores simulados.
- [Variantes `#[non_exhaustive}` exigem caso coringa em matches externos] → intencional: é o preço da evolução não-breaking; documentado como padrão de consumo.
- [Decisões adiadas podem custar retrabalho localizado] → lista explícita abaixo com direção padrão; todas são internas, sem impacto no contrato público.

## Decisões adiadas conscientemente (com direção padrão)

Representação interna da variante nominal (direção: enum interno minúsculo aceitando fatia de referências e fatia de strings owned, exposto como variante única pública); multi-alvo/multi-rótulo 2D (adiado — nova variante futura, não-breaking); conversões curadas com feature flags para `chrono`/`uuid`/`arrow` e macro `#[derive(SKTarget)]` (changes futuras); amarração fina entre plano de execução e escolha de alocadores (chega com os allocators).

## Migration Plan

Crate novo; sem migração. Consumidores futuros começam contra esta API; rollback = reverter merge da branch de código.

## Open Questions

Nenhuma que bloqueie o detalhamento de tarefas — as micro-decisões listadas acima têm direção padrão definida e serão confirmadas durante o TDD de cada módulo.
