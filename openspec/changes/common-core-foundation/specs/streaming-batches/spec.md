## Purpose

Define o contrato de streaming out-of-core do PRD (§4.4): blocos de dados owned com metadados mínimos, atravessáveis entre threads para permitir o pipeline I/O ∥ CPU, e as duas fontes complementares — sequencial por iteração e acesso aleatório abstrato (memmap chega na interop).

## ADDED Requirements

### Requirement: Batches owned com metadados mínimos
Cada bloco de streaming SHALL possuir integralmente seus dados (sem empréstimo da fonte), carregando posição na sequência e indicação de bloco final, de modo a poder ser movido para outra thread enquanto a fonte avança.

#### Scenario: Bloco sobrevive à fonte
- **WHEN** um bloco é extraído do iterador da fonte e a referência ao iterador é descartada em seguida
- **THEN** os dados do bloco permanecem íntegros e utilizáveis — propriedade que habilita processamento assíncrono fora da thread de leitura

#### Scenario: Último bloco é identificável
- **WHEN** uma fonte finita é consumida até o fim
- **THEN** exatamente um bloco é marcado como final

### Requirement: Fonte sequencial como iterador falível
A fonte de streaming sequencial SHALL expor-se como iteração de blocos com erros da taxonomia central, permitindo falhas de leitura intermediárias sem pânico.

#### Scenario: Falha de leitura interrompe com erro estruturado
- **WHEN** a leitura de um bloco intermediário falha
- **THEN** a iteração produz o erro da taxonomia central e o consumidor decide encerrar ou tratar

### Requirement: Fonte de acesso aleatório abstrata
A biblioteca SHALL definir o contrato de fonte com acesso posicional direto às unidades de dados, independente do mecanismo de armazenamento; implementações concretas mapeadas em memória pertencem à camada de interop.

#### Scenario: Acesso posicional direto sem varredura
- **WHEN** uma unidade arbitrária é solicitada pelo índice a uma fonte de acesso aleatório
- **THEN** o acesso não exige percorrer unidades anteriores

#### Scenario: Contrato não acopla mecanismo de armazenamento
- **WHEN** um provedor implementa a fonte sobre qualquer mecanismo persistente próprio
- **THEN** nenhuma dependência específica de mapeamento de memória é exigida pela definição do contrato
