## Purpose

Define a fronteira de dados da biblioteca: as representações canônicas de features (densa/esparsa) e de alvos (contínuo/inteiro/nominal), o mecanismo universal de conversão aceito nas entradas públicas, a garantia zero-copy das conversões nativas, a evolução não-breaking por novas variantes e a canonicalização de rótulos que sustenta classificadores e codecs.

## ADDED Requirements

### Requirement: View de features densa/esparsa como enum extensível
A biblioteca SHALL expor um único tipo de view de features cobrindo representação densa e esparsa, marcado como não exaustivo para permitir novas variantes sem quebra de consumidores que usam correspondência com caso coringa.

#### Scenario: Matriz densa emprestada entra sem cópia
- **WHEN** uma matriz densa existente é passada como view emprestada a uma operação
- **THEN** nenhuma cópia dos elementos é realizada pela conversão

#### Scenario: Matriz esparsa entra pela mesma fronteira
- **WHEN** uma matriz esparsa no formato comprimido por linhas é passada a uma operação
- **THEN** a conversão produz a variante esparsa da view referenciando os mesmos dados

#### Scenario: Bloco owned intermediário entra por empréstimo
- **WHEN** o resultado owned de um estágio anterior de pipeline é passado ao estágio seguinte
- **THEN** a conversão empresta o bloco sem duplicar seus dados

### Requirement: Seam universal de conversão nas entradas públicas
Toda entrada pública de dados SHALL ser declarada sobre o trait de conversão falível da linguagem padrão, de modo que: tipos com conversão infalível sejam aceitos automaticamente via implementação em blanket da biblioteca padrão, e integradores externos possam plugar tipos próprios implementando conversões — inclusive falíveis — para os tipos de view da biblioteca.

#### Scenario: Tipo do usuário com Into funciona sem atrito
- **WHEN** um consumidor passa à API um tipo que implementa apenas conversão infalível para a view
- **THEN** a chamada compila e executa sem código adicional, pois a std promove automaticamente essa conversão à forma falível exigida pelo bound

#### Scenario: Conversão falível de terceiro reporta erro estruturado
- **WHEN** um integrador externo implementa conversão falível de seu tipo para a view e ela falha durante uma chamada
- **THEN** o erro chega ao consumidor através do Result da própria operação chamada, na variante de falha de conversão, sem pânico

### Requirement: Rejeição precisa de representação não suportada
Algoritmos que suportam apenas parte das representações SHALL rejeitar as demais em runtime com erro específico de representação não suportada, orientando o consumidor à conversão adequada; o despacho de representação SHALL ocorrer uma única vez por operação, nunca por elemento.

#### Scenario: Esparsa rejeitada por algoritmo denso-only
- **WHEN** dados esparsos são entregues a um algoritmo que declara suportar apenas densos
- **THEN** a operação falha antes de processar qualquer elemento, com erro indicando a incompatibilidade e o caminho de conversão sugerido

### Requirement: View de alvos contínua/inteira/nominal
A biblioteca SHALL expor um único tipo de view de alvos com três representações — valores contínuos, inteiros e nominais (símbolos textuais) — marcado como não exaustivo; inteiros SHALL poder ser elevados a contínuos de forma lossless quando a interpretação contínua for legítima.

#### Scenario: Alvos inteiros elevados para regressão
- **WHEN** alvos armazenados como inteiros são fornecidos a um contexto de leitura contínua
- **THEN** a elevação para ponto flutuante ocorre sem perda e sem intervenção manual

#### Scenario: Alvos nominais referenciam texto emprestado
- **WHEN** rótulos textuais são fornecidos como alvos
- **THEN** a view nominal referencia as strings originais sem copiá-las

### Requirement: Canonicalização determinística de rótulos
A biblioteca SHALL oferecer canonicalização de alvos nominais/inteiros/booleanos para índices compactos acompanhados de tabela reversível, determinística para a mesma sequência de entrada, servindo de base à codificação automática de classificadores e aos codecs explícitos futuros.

#### Scenario: Roundtrip preserva rótulos originais
- **WHEN** uma sequência de rótulos é canonicalizada e os índices resultantes são decodificados pela tabela produzida
- **THEN** a sequência original de rótulos é restituída integralmente

#### Scenario: Mesma entrada produz mesma tabela
- **WHEN** a mesma sequência de rótulos é canonicalizada duas vezes
- **THEN** os mapeamentos rótulo→índice produzidos são idênticos entre si
