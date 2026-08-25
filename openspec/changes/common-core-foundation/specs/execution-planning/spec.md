## Purpose

Define o mecanismo de decisão de modo de execução do PRD (§5.3): separação entre intenção declarada pelo usuário e plano resolvido por operação, resolução pura e determinística com contexto injetável, e falha dura para pedidos explícitos incompatíveis com o padrão de acesso do algoritmo.

## ADDED Requirements

### Requirement: Intenção e plano são conceitos distintos
A biblioteca SHALL distinguir a intenção de execução declarada pelo consumidor — incluindo o modo automático como padrão — do plano efetivo resolvido no momento da operação, que consolida modo escolhido, paralelismo e tamanho de lote.

#### Scenario: Intenção automática produz plano concreto
- **WHEN** uma operação executa com intenção automática sobre um contexto conhecido
- **THEN** um plano concreto é produzido antes do processamento dos dados

#### Scenario: Intenção explícita preservada quando compatível
- **WHEN** o consumidor declara explicitamente um modo compatível com o padrão de acesso do algoritmo
- **THEN** o plano resolvido reflete exatamente esse modo

### Requirement: Resolução pura, determinística e injetável
A resolução SHALL ser função pura sobre a intenção e um contexto explícito — memória disponível, núcleos de CPU, tamanho do conjunto, padrão de acesso declarado pelo algoritmo e dica de lote opcional — sem acesso a estado global; o contexto SHALL poder ser fornecido pelo chamador, permitindo testes determinísticos independentes da máquina.

#### Scenario: Mesmo contexto produz mesmo plano
- **WHEN** a resolução é executada duas vezes com intenção e contexto idênticos
- **THEN** os planos resultantes são idênticos em todos os campos

#### Scenario: Contexto simulado dispensa máquina real
- **WHEN** os testes exercitam a resolução com valores simulados de memória e núcleos
- **THEN** os planos obtidos refletem exclusivamente os valores simulados, sem leitura do ambiente físico

### Requirement: Falha dura para incompatibilidade explícita
Pedido explícito de modo incompatível com o padrão de acesso declarado pelo algoritmo SHALL falhar com erro específico nomeando o modo pedido e o padrão declarado, verificado antes do processamento de qualquer dado; o modo automático SHALL NUNCA produzir tal erro.

#### Scenario: Streaming sequencial recusado para algoritmo de acesso aleatório
- **WHEN** o consumidor pede explicitamente streaming sequencial a um algoritmo cujo padrão declarado é acesso aleatório
- **THEN** a operação falha imediatamente com o erro de incompatibilidade, identificando ambos os lados do conflito

#### Scenario: Automático nunca conflita com o padrão declarado
- **WHEN** a intenção é automática, quaisquer que sejam contexto e algoritmo
- **THEN** a resolução sempre escolhe modo compatível com o padrão declarado e não falha por incompatibilidade

### Requirement: Resolução ocorre por operação
Cada operação pesada SHALL resolver seu próprio plano a partir da intenção armazenada e do contexto daquele momento — ajuste e predição resolvem independentemente, pois o tamanho do dado só se torna conhecido na entrada de cada operação.

#### Scenario: Predição sobre volume maior que o ajuste resolve próprio plano
- **WHEN** o ajuste processa um conjunto pequeno em memória e a predição posterior recebe volume que excede a memória disponível simulada
- **THEN** o plano da predição difere do plano do ajuste, refletindo o novo contexto sem reconfiguração manual
