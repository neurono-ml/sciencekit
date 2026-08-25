## Purpose

Define os contratos de ajuste e transformação da biblioteca: duas traits separadas por eixo de supervisão, separação tipada entre estimador configurado e modelo ajustado (imutável, compartilhável entre threads), impossibilidade em compilação de prever sem ajustar, e transformadores com saída tipada por associated type — requisito do encadeamento estático de pipelines.

## ADDED Requirements

### Requirement: Duas traits de ajuste por supervisão
A biblioteca SHALL expor uma trait de ajuste não supervisionado que recebe apenas features e uma trait de ajuste supervisionado que recebe features e alvos; estimadores SHALL implementar exclusivamente a(s) trait(s) compatível(eis) com sua natureza.

#### Scenario: Classificador exige alvos
- **WHEN** um estimador supervisionado é usado
- **THEN** seu método de ajuste exige alvos na assinatura, e chamadas sem alvos não compilam

#### Scenario: Clusterizador recusa alvos em compilação
- **WHEN** código tenta fornecer alvos ao ajuste de um estimador que só implementa a trait não supervisionada
- **THEN** a compilação falha

### Requirement: Estimador configurado é separado do modelo ajustado
O resultado do ajuste SHALL ser um tipo distinto — o modelo — portador exclusivo do estado aprendido; o ajuste SHALL operar sobre referência compartilhada do estimador configurado, preservando-o inalterado e reutilizável para novos ajustes, inclusive simultâneos.

#### Scenario: Mesmo estimador alimenta ajustes paralelos
- **WHEN** o mesmo estimador configurado é usado como origem de múltiplos ajustes concorrentes sobre partições distintas
- **THEN** todos os ajustes progridem sem exclusão mútua sobre o estimador e cada um produz seu próprio modelo

#### Scenario: Ajuste repetido com hiperparâmetros idênticos é determinístico em interface
- **WHEN** o mesmo estimador configurado ajusta duas vezes os mesmos dados determinísticos
- **THEN** ambos os modelos resultantes são instâncias independentes do mesmo tipo de modelo

### Requirement: Prever antes de ajustar é irrepresentável
Métodos de predição SHALL existir apenas no tipo de modelo ajustado; o tipo de estimador configurado SHALL NOT expor predição, tornando uso incorreto um erro de compilação.

#### Scenario: Predição sobre estimador não ajustado não compila
- **WHEN** código tenta chamar predição diretamente sobre o estimador configurado
- **THEN** a compilação falha por ausência do método nesse tipo

### Requirement: Modelos ajustados são compartilháveis entre threads
Tipos de modelos ajustados SHALL satisfazer envio e compartilhamento seguro entre threads por construção, sem mutex externo para leitura concorrente.

#### Scenario: Um modelo atende múltiplas threads simultaneamente
- **WHEN** um mesmo modelo ajustado é compartilhado entre threads que executam predições concorrentes
- **THEN** todas as predições completam sem sincronização adicional exigida do consumidor

### Requirement: Transformador com saída tipada
A trait de transformação SHALL declarar o tipo produzido pela transformação como associated type, permitindo que consumidores e pipelines futuros validem estaticamente a compatibilidade entre a saída de um estágio e a entrada do seguinte.

#### Scenario: Encadeamento compatível valida estaticamente
- **WHEN** um pipeline conecta a saída declarada de um transformador à entrada de outro estágio via conversão padrão aceita pela fronteira de dados
- **THEN** o encadeamento compila sem verificação em runtime de tipos intermediários

#### Scenario: Incompatibilidade declarada falha cedo
- **WHEN** o tipo de saída de um transformador não pode converter-se à representação exigida pelo estágio seguinte
- **THEN** a incompatibilidade é detectável em tempo de compilação pelo consumidor do contrato
