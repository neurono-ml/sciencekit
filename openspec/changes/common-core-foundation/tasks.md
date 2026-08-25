# Tasks — common-core-foundation

Toda task segue TDD (skill `tdd`): teste primeiro em módulo companion `*_tests.rs`, falha confirmada, implementação mínima, refatoração. Nenhum arquivo além de 200 linhas — estrutura de módulo pasta já prevista no design.

## 1. Estrutura do crate

- [ ] 1.1 Criar `crates/sciencekit_common` no workspace com manifesto, licença declarada e árvore de módulos pasta vazia (`sk_float/`, `errors/`, `data_view/`, `target_view/`, `label_table/`, `fit_traits/`, `scorer_traits/`, `execution/`, `batching/`), compilando verde
- [ ] 1.2 Adicionar dependências base: `ndarray`, `sprs`, `num-traits`, `thiserror`, `sysinfo`

## 2. Tipagem escalar

- [ ] 2.1 TDD da trait selada `SKFloat`: implementada pelos floats padrão suportados, não implementável externamente, inteiros rejeitados por bound

## 3. Modelo de erros

- [ ] 3.1 TDD do enum central com variantes acordadas (forma com esperado/encontrado, representação não suportada, hiperparâmetro inválido identificável, modo incompatível com padrão declarado, não convergência com contagem, E/S com causa preservada, falha de conversão)
- [ ] 3.2 TDD das conversões automáticas a partir de erro de plataforma e verificação do padrão de derivação para enums futuros de algoritmo

## 4. Fronteira de dados

- [ ] 4.1 TDD da view densa/esparsa: conversões nativas sem cópia (denso emprestado, esparso CSR, bloco owned por empréstimo), marcação não exaustiva
- [ ] 4.2 TDD do seam falível: tipos nativos via conversões diretas, tipo com conversão infalível aceito via blanket da std, conversão falível de terceiro propagando erro estruturado ao Result da operação
- [ ] 4.3 TDD da view de alvos: três variantes (contínua, inteira, nominal), elevação lossless inteiro→contínuo, nominal referenciando texto emprestado
- [ ] 4.4 TDD da rejeição precisa: variante não suportada por um consumidor denso-only produz erro específico antes de processar elementos

## 5. Rótulos canônicos

- [ ] 5.1 TDD da canonicalização determinística: sequências nominais/inteiras → índices compactos + tabela reversível; roundtrip integral; mesma entrada → mesmo mapeamento
- [ ] 5.2 TDD da tabela como portadora de dados exportáveis (metadados legíveis para o futuro header de modelos)

## 6. Contratos de ajuste e transformação

- [ ] 6.1 TDD da trait não supervisionada: recebe apenas features; retorno é associated type do modelo; ajuste sobre referência compartilhada preserva o estimador reutilizável
- [ ] 6.2 TDD da trait supervisionada: exige alvos na assinatura; estimador que só implementa a não supervisionada recusa alvos em compilação
- [ ] 6.3 Testes de compilação provando irrepresentabilidade de prever-sem-ajustar e `Send`/`Sync` dos modelos produzidos por contratos de exemplo
- [ ] 6.4 TDD da trait de transformador com saída tipada: encadeamento compatível valida estaticamente; incompatibilidade declarada detectável em compilação

## 7. Contratos de pontuação

- [ ] 7.1 TDD do scorer supervisionado: forma pura sobre previsões existentes sem inferência; forma provida executa inferência sobre contrato de exemplo e delega, resultado equivalente
- [ ] 7.2 TDD do scorer não supervisionado: forma pura sobre atribuições existentes + forma provida análoga
- [ ] 7.3 TDD da falibilidade: entradas incomparáveis produzem erro estruturado da taxonomia; genericidade do modelo avaliado permite reuso entre famílias

## 8. Planejamento de execução

- [ ] 8.1 TDD da enum de intenção (cinco modos) e struct de plano consolidado
- [ ] 8.2 TDD da resolução pura: mesmo contexto → mesmo plano; intenção explícita compatível preservada no plano
- [ ] 8.3 TDD do erro duro: modo explícito incompatível com padrão declarado falha nomeando ambos os lados, antes de processar dados; automático nunca produz esse erro
- [ ] 8.4 TDD da resolução por operação: contextos distintos entre ajuste e predição produzem planos independentes; contexto simulado dispensa leitura física (construtor padrão confinado à leitura real)

## 9. Streaming

- [ ] 9.1 TDD do bloco owned com metadados: sobrevive ao descarte da fonte; exatamente um bloco final em fonte finita
- [ ] 9.2 TDD da fonte sequencial: iteração falível com erro estruturado em falha intermediária
- [ ] 9.3 TDD do contrato de acesso aleatório abstrato: acesso posicional direto sem varredura, sem acoplar mecanismo de armazenamento

## 10. Aceite e revisão

- [ ] 10.1 Rodar todos os gates locais (fmt, clippy estrito, testes, doctests) e confirmar verde
- [ ] 10.2 Verificar checklist de aceite adaptado desta change: contratos compilam sob `Send`/`Sync` onde prometido, coberturas companion completas, dados mock em ndarray/sprs, nomenclatura completa com prefixos corretos, nenhum arquivo além de 200 linhas
- [ ] 10.3 Registrar no PR os critérios de aceite completos do PRD §8.7/§10.3 como pendentes para o primeiro estimador (Fase 1)
