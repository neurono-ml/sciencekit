# Spec Delta

## ADDED Requirements

### Requirement: Language tab widget for grouped code blocks

The book theme SHALL render a group of consecutive code blocks that carry the same group
identifier as a single tabbed widget, with one tab per language, the first language
selected by default, and keyboard-accessible tab controls. The widget SHALL degrade to
plain code blocks when JavaScript is unavailable.

#### Scenario: Tabs select a language

- **WHEN** a reader selects the second tab of a grouped code widget
- **THEN** the code block of the second language becomes visible and the first one is
  hidden

#### Scenario: Keyboard navigation

- **WHEN** a reader focuses the tab list and presses the arrow keys
- **THEN** the focus moves between the language tabs and the selection follows

#### Scenario: No JavaScript

- **WHEN** the page is rendered without JavaScript
- **THEN** every code block of the group is visible and readable as plain code

### Requirement: Gallery navigation for the Examples section

The book theme SHALL provide navigation for the Examples section that lets a reader
browse the examples by algorithm family, and the sidebar SHALL expose the family pages as
their own collapsible section.

#### Scenario: Family pages are grouped in the sidebar

- **WHEN** the reader opens the sidebar of a book containing the Examples section
- **THEN** the family pages appear grouped under the Examples section, each family
  collapsible

#### Scenario: Current page is exposed

- **WHEN** a reader opens a gallery page
- **THEN** its family section is expanded and the page is marked as active

### Requirement: Cross-book link styling

Links that point from the main book to the separate machine-learning book, and links
between an example page and a chapter, SHALL be visually distinguishable from ordinary
internal links and SHALL state where they lead.

#### Scenario: Cross-book link is distinguishable

- **WHEN** a reader encounters a link to the separate machine-learning book
- **THEN** the link is styled as an external-to-the-book link and states the target
  book
