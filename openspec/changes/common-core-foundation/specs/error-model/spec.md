## Purpose

Estabelece a taxonomia central de erros compartilhada por toda a biblioteca: variantes comuns nomeadas e precisas, conversão uniforme para erros específicos de algoritmo e ausência de tipos de erro ad hoc nos contratos centrais.

## ADDED Requirements

### Requirement: Erro central com variantes acordadas
A biblioteca SHALL expor um enum central de erro cobrindo, no mínimo: incompatibilidade de forma (com dimensões esperadas e encontradas), representação de dados não suportada pelo algoritmo, hiperparâmetro inválido (identificável), modo de execução incompatível com o padrão de acesso declarado, não convergência (com contagem de iterações executadas), erro de entrada/saída e falha de conversão na fronteira de dados.

#### Scenario: Incompatibilidade de forma identifica dimensões
- **WHEN** uma operação recebe dados cujas dimensões não são compatíveis com a operação
- **THEN** o erro produzido é a variante de forma, carregando a forma esperada e a recebida

#### Scenario: Representação não suportada é distinguida de forma inválida
- **WHEN** um algoritmo que opera apenas sobre dados densos recebe representação esparsa
- **THEN** o erro produzido é especificamente de representação não suportada — e não um erro genérico de forma

#### Scenario: Não convergência informa esforço realizado
- **WHEN** um processo iterativo esgota as iterações sem convergir
- **THEN** o erro produzido informa a quantidade de iterações executadas

### Requirement: Conversão uniforme para erros de algoritmo
Cada crate de algoritmo SHALL poder definir seu próprio tipo de erro, e esse tipo SHALL converter-se a partir do erro central automaticamente, preservando os erros comuns idênticos entre algoritmos.

#### Scenario: Erro central propaga através de erro de algoritmo
- **WHEN** um consumidor trabalha com o tipo de erro específico de um algoritmo e ocorre um erro comum da biblioteca dentro do fluxo desse algoritmo
- **THEN** o erro comum é convertido automaticamente para o tipo do algoritmo via conversão padrão da linguagem

### Requirement: Erros de E/S integram-se à taxonomia
Erros de entrada/saída da plataforma SHALL converter-se ao erro central sem envolvimento manual do chamador.

#### Scenario: Falha de I/S vira erro da biblioteca
- **WHEN** uma operação de leitura/escrita falha com erro de plataforma durante processamento
- **THEN** o consumidor recebe o erro central na variante de E/S, com o erro original preservado como causa
