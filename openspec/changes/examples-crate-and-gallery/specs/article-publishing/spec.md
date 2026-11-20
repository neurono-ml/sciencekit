# Spec Delta

## Purpose

Defines the article ecosystem published outside the book: one Medium article and one
LinkedIn article per chapter, each self-contained, optimized for its platform and
linked back to the official page it was derived from.

## ADDED Requirements

### Requirement: Every chapter has an article for each platform

Every chapter of the book SHALL have a Medium article and a LinkedIn article produced
by the repository, and the articles SHALL be regenerated from the chapter source rather
than written by hand.

#### Scenario: The articles are generated

- **WHEN** the article pipeline runs
- **THEN** it produces one Medium article and one LinkedIn article per chapter from the
  chapter's source

#### Scenario: Regeneration is deterministic

- **WHEN** the pipeline runs twice on an unchanged chapter
- **THEN** the produced articles are identical

### Requirement: An article is self-contained

An article SHALL carry the substance of its chapter — the explanation, the code, the
tables and the figures — so that a reader who arrives from the platform's search or
feed can follow it without opening the book.

#### Scenario: The example survives the move

- **WHEN** an article is read on its own
- **THEN** the example it presents is complete enough to be understood and run

#### Scenario: The code is present

- **WHEN** an article contains an example
- **THEN** it includes the code blocks and the explanation of what they show

### Requirement: An article points back to the official page

Every article SHALL link to the canonical page of the chapter it derives from, SHALL
link to the chapter before and the chapter after it, and SHALL link to the book's table
of contents.

#### Scenario: The canonical link is present

- **WHEN** an article is read
- **THEN** it names the official page it derives from, and the link is the canonical
  one

#### Scenario: The navigation is present

- **WHEN** an article is read
- **THEN** it links to the previous chapter's article, to the next chapter's article
  and to the book's table of contents

### Requirement: An article is optimized for its platform

A Medium article SHALL carry a platform-appropriate title, subtitle and tag set and
SHALL be readable as a standalone essay; a LinkedIn article SHALL carry a
platform-appropriate headline and a short opening that states the value for the
professional reader, and SHALL be published in the ecosystem of the other LinkedIn
articles.

#### Scenario: The Medium variant is an essay

- **WHEN** the Medium article is read
- **THEN** it opens with the problem the chapter solves, and it reads as a complete
  piece rather than as a manual page

#### Scenario: The LinkedIn variant addresses the professional

- **WHEN** the LinkedIn article is read
- **THEN** it opens with the professional consequence of the subject and states what
  the reader will be able to do afterwards

#### Scenario: The LinkedIn articles form an ecosystem

- **WHEN** the LinkedIn articles are listed
- **THEN** each one links to the others and to the series index, so the reader can
  follow the series in order

### Requirement: The articles declare their origin

Every article SHALL state that it is derived from the official book, SHALL name the
chapter it comes from, and SHALL be discoverable by the platform's search through its
own title, description and tags.

#### Scenario: The origin is stated

- **WHEN** an article is read
- **THEN** its opening states that the content comes from the official book and names
  the chapter

#### Scenario: The article is discoverable

- **WHEN** the article's metadata is inspected
- **THEN** it carries a title, a description and tags suited to its platform
