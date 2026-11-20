# Spec Delta

## Purpose

Defines the formatting contract of the book: standard mdBook rendering, semantic
markdown, a minimal skin, and the rules that keep every page readable and printable.

## ADDED Requirements

### Requirement: The book renders with the standard mdBook formatting

The book SHALL be rendered by mdBook with the default theme and with a stylesheet that
extends the default one rather than replacing it, and SHALL NOT depend on bespoke
markup to convey content.

#### Scenario: The default theme is legible

- **WHEN** the book is built and a page is read
- **THEN** headings, paragraphs, lists, tables, block quotes and code blocks render
  with the default mdBook appearance

#### Scenario: No bespoke markup is required to read a page

- **WHEN** a page is read with the stylesheet disabled
- **THEN** its content, its structure and its code examples remain complete and
  understandable

### Requirement: The pages are semantic markdown

Every page SHALL be authored in markdown with a single level-one heading, a hierarchy
of headings that never skips a level, and no raw HTML used to lay out content.

#### Scenario: The heading hierarchy is sound

- **WHEN** the headings of a page are listed
- **THEN** the page starts with one level-one heading and no level is skipped

#### Scenario: Content is not laid out with HTML

- **WHEN** the source of a page is inspected
- **THEN** the content is markdown and the raw HTML is limited to elements that carry
  meaning, such as figures and generated metadata

### Requirement: The skin is minimal

The book's stylesheet SHALL contain only the rules that mdBook itself does not provide:
code-block sizing, the language tabs, the Mermaid and mathematics containers, print
rules and accessibility affordances.

#### Scenario: The stylesheet is small

- **WHEN** the stylesheet is measured
- **THEN** it is a small fraction of the length it had before this change, and every
  rule it keeps has a stated purpose

#### Scenario: The theme can be switched

- **WHEN** the reader switches the colour theme
- **THEN** every page remains readable, including the diagrams and the code blocks

### Requirement: Code examples are runnable and labelled

Every code block in the book SHALL state the language it is written in, SHALL be a
complete, runnable fragment, and the shell commands it shows SHALL be copy-pasteable.

#### Scenario: The commands work as written

- **WHEN** a shell command from the book is copied and run
- **THEN** it runs as written in the environment the chapter describes

#### Scenario: The examples carry their input

- **WHEN** a code example is read
- **THEN** it states where its data comes from and how the reader obtains it
