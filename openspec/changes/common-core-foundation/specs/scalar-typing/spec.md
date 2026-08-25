## Purpose

Define o contrato numérico único da biblioteca: uma trait selada que reúne as garantias exigidas de tipos de ponto flutuante usados em todas as APIs genéricas, permitindo dtypes genéricos por parâmetro de trait sem abrir a superfície a implementações externas arbitrárias.

## ADDED Requirements

### Requirement: Trait numérica selada única
A biblioteca SHALL expor exatamente uma trait selada de ponto flutuante agregando os bounds necessários para computação numérica (aritmética, cópia, envio entre threads e uso estático), e toda API genérica que aceite dados numéricos contínuos SHALL usar essa trait como bound — nunca bounds soltos espalhados.

#### Scenario: Tipos float nativos satisfazem o contrato
- **WHEN** um algoritmo genérico é instanciado com os tipos de ponto flutuante padrão suportados pela biblioteca
- **THEN** a instanciação compila sem implementações adicionais do usuário

#### Scenario: Implementação externa é impedida
- **WHEN** um crate externo tenta implementar a trait numérica para um tipo próprio
- **THEN** a compilação falha porque a trait é selada

### Requirement: Inteiros não satisfazem contratos de contínuos
Tipos inteiros SHALL NOT satisfazer o bound numérico de APIs contínuas; dados inteiros entram na biblioteca pelas representações próprias de alvos/integrais definidas na fronteira de dados.

#### Scenario: Uso de inteiro onde se espera contínuo falha em compilação
- **WHEN** código tenta instanciar um estimador contínuo com tipo inteiro como dtype
- **THEN** a compilação falha com violação de bound
