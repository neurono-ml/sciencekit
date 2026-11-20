# Spec Delta

## Purpose

Defines the search-engine contract of the book: what a crawler and a reader's search
result must find on every page, and how the book describes itself to Google and Bing.

## ADDED Requirements

### Requirement: Every page declares its identity

Every rendered page SHALL carry a title, a description, keywords, a canonical URL and
the Open Graph and Twitter card properties derived from those.

#### Scenario: The metadata is present

- **WHEN** a rendered page is inspected
- **THEN** it carries a title, a meta description, keywords, a canonical link, an
  Open Graph title and description and a Twitter card

#### Scenario: The description is written for a search result

- **WHEN** a page's description is read
- **THEN** it is one or two sentences that describe the page without repeating its
  title

### Requirement: The pages declare their structure to the engines

Every page SHALL carry structured data describing the chapter and the breadcrumb path
that reaches it, and the book SHALL ship a sitemap listing every page and a robots
policy that permits the pages and points at the sitemap.

#### Scenario: The structured data is valid

- **WHEN** the structured data of a page is validated
- **THEN** it describes the chapter, its position in the book and its canonical URL

#### Scenario: The sitemap is complete

- **WHEN** the sitemap is compared with the rendered pages
- **THEN** every page appears exactly once, and the robots policy points at it

### Requirement: The pages are internally linked

Every chapter SHALL link to the chapters that precede and follow it, to the table of
contents, and to the examples it uses, and the book SHALL NOT contain a page that
cannot be reached from the table of contents.

#### Scenario: The navigation is present

- **WHEN** a chapter is read
- **THEN** it links to the previous chapter, to the next chapter and to the table of
  contents

#### Scenario: No page is orphaned

- **WHEN** the link graph of the book is built
- **THEN** every page is reachable from the table of contents

### Requirement: The pages are written for the reader who arrives from a search

A chapter that a reader can land on directly SHALL state, in its opening, what it
covers, and SHALL NOT depend on the previous chapter to be understood.

#### Scenario: A chapter stands alone

- **WHEN** a chapter is read without its predecessors
- **THEN** it explains its own subject, states its prerequisites and can be followed
  on its own
