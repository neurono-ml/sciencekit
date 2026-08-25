## Purpose

Define os contratos de avaliação: scorers supervisionados e não supervisionados com dupla entrada — forma pura sobre previsões/atribuições já existentes (sem re-inferência) e forma conveniente provida que executa inferência e delega — ambos falíveis, servindo de base a GridSearchCV e pipelines.

## ADDED Requirements

### Requirement: Scorer supervisionado com forma pura e forma conveniente
O scorer supervisionado SHALL expor uma entrada pura que compara alvos verdadeiros com previsões já existentes sem executar inferência, e SHALL prover uma segunda entrada que recebe modelo, features e alvos verdadeiros, executa a inferência e delega à forma pura — sem exigir implementação adicional do autor do scorer.

#### Scenario: Métrica sobre previsões armazenadas não reinferi
- **WHEN** o consumidor já dispõe das previsões de um modelo e avalia via forma pura
- **THEN** nenhuma inferência é executada e o valor da métrica é produzido diretamente da comparação

#### Scenario: Forma conveniente executa inferência e delega
- **WHEN** um consumidor avalia um modelo via forma conveniente
- **THEN** as previsões são obtidas do modelo e o resultado equivale ao da forma pura aplicada a essas mesmas previsões

### Requirement: Scorer não supervisionado por atribuições ou por modelo
O scorer não supervisionado SHALL expor entrada pura sobre features e atribuições/saídas já existentes (ex.: rótulos de cluster) e SHALL prover entrada conveniente análoga que obtém as saídas do modelo antes de delegar.

#### Scenario: Silhueta-like sobre atribuições existentes
- **WHEN** atribuições de cluster já calculadas são fornecidas com suas features à forma pura
- **THEN** a pontuação é calculada sem contato com o modelo

### Requirement: Avaliação é falível por construção
Ambos os contratos de pontuação SHALL retornar resultado falível, pois a forma conveniente pode falhar na inferência e as formas puras podem rejeitar entradas incoerentes.

#### Scenario: Previsões incoerentes com os alvos produzem erro estruturado
- **WHEN** a forma pura recebe previsões cuja estrutura não é comparável aos alvos verdadeiros
- **THEN** a operação retorna erro da taxonomia central — nunca entra em pânico nem devolve sentinela numérica

### Requirement: Scorers são independentes dos modelos que avaliam
Os contratos de pontuação SHALL ser genéricos quanto ao modelo avaliado, permitindo scorers reutilizáveis entre famílias de algoritmos compatíveis com a mesma forma de saída.

#### Scenario: Mesmo scorer avalia modelos de famílias distintas
- **WHEN** dois modelos supervisionados de naturezas diferentes produzem previsões comparáveis aos mesmos alvos
- **THEN** o mesmo scorer os avalia sem adaptação adicional
