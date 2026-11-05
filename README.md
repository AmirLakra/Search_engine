# Text Search Engine

A lightweight desktop text-search application built with **Python** and
**Tkinter**. It reads a plain-text corpus, splits it into searchable
document units, builds in-memory indexes, evaluates search queries,
ranks matching documents, and displays the results in a graphical
interface.

The project demonstrates core information-retrieval concepts---including
tokenization, stemming, inverted indexes, Boolean query processing,
wildcard matching, phonetic lookup, edit distance, and TF-IDF-style
scoring---without depending on an external search platform.

------------------------------------------------------------------------

## Features

-   **TXT corpus loading** through a graphical file picker.
-   **Document segmentation** by blank-line-separated paragraphs, with a
    non-empty-line fallback.
-   **Tokenization** using regular expressions, retaining token
    positions and character offsets.
-   **Text normalization** with lowercasing, stop-word filtering, and
    stemming.
-   **Positional inverted index** mapping terms to document IDs and
    token positions.
-   **K-gram index** for wildcard candidate generation and
    spelling-correction candidates.
-   **Boolean search** with `AND`, `OR`, `NOT`, parentheses, and
    implicit `AND` between adjacent terms.
-   **Phrase search** using double quotes.
-   **Wildcard search** with `*` and `?`.
-   **Soundex search** using `soundex:word` or `~word`.
-   **Automatic spelling correction** for some single-term queries when
    a close vocabulary candidate is found.
-   **TF-IDF-style ranking** of matching documents.
-   **Tkinter GUI** with suggested searches, a scrollable results panel,
    and highlighted matching terms.

## Technology Stack

  Technology                  Purpose
  --------------------------- ------------------------------------------
  Python 3                    Main programming language
  Tkinter / ttk               Desktop graphical interface
  `re`                        Tokenization and query token recognition
  `collections.defaultdict`   In-memory index structures
  `math`                      Logarithmic relevance scoring
  NLTK (optional)             Porter stemming when installed

The application does not require a database, web server, or third-party
search engine. NLTK is optional; a lightweight built-in suffix stemmer
is used when NLTK is unavailable.

------------------------------------------------------------------------

## Requirements

-   Python 3
-   Tkinter (usually included with standard Python distributions)
-   Optional: NLTK for Porter stemming

### Install the optional dependency

``` bash
pip install nltk
```

If you do not install NLTK, the application uses its built-in fallback
stemmer.

------------------------------------------------------------------------

## Run the Application

1.  Clone or download this project.
2.  Open a terminal in the project directory.
3.  Run:

``` bash
python search_engine.py
```

4.  In the application, select **Load text file** and choose a `.txt`
    file.
5.  Enter a query in the search field and press **Enter** or click the
    search arrow.
6.  Review the ranked results in the results panel.

> The application indexes the selected text in memory. Loading another
> file rebuilds the indexes for the newly selected corpus.

------------------------------------------------------------------------

## Search Query Guide

  -----------------------------------------------------------------------
  Query                               Description
  ----------------------------------- -----------------------------------
  `cat`                               Find documents containing `cat`.

  `cat AND dog`                       Find documents containing both
                                      terms.

  `cat OR dog`                        Find documents containing either
                                      term.

  `cat AND NOT dog`                   Find documents containing `cat`
                                      while excluding documents
                                      containing `dog`.

  `(cat OR dog) AND NOT machine`      Combine Boolean operators using
                                      parentheses.

  `"machine learning"`                Search for the adjacent phrase
                                      `machine learning`.

  `information retrieval`             Adjacent terms are treated as an
                                      implicit `AND`.

  `comp*`                             Use `*` to match zero or more
                                      characters.

  `c?t`                               Use `?` to match one character.

  `soundex:smythe`                    Find indexed terms with the same
                                      Soundex code as `smythe`.

  `~smythe`                           Alternative Soundex query notation.

  `informaton`                        Try automatic correction if a
                                      sufficiently close indexed term is
                                      found.
  -----------------------------------------------------------------------

Boolean operators are recognized without regard to case. Use parentheses
to make grouped expressions explicit.

### Query processing notes

-   The parser evaluates `NOT` at the atom level, `AND` before `OR`, and
    parentheses as grouped expressions.
-   Stop words are omitted from the inverted index. A query consisting
    only of a stop word therefore does not retrieve documents.
-   Automatic correction is a best-effort feature with fixed similarity
    and edit-distance thresholds. It may not correct every misspelling.
-   Wildcard expansion depends on the terms generated from the corpus
    vocabulary and the k-gram candidate index.

------------------------------------------------------------------------

## How It Works

### 1. Load and split the corpus

The selected file is read as text. The application splits it into
paragraphs separated by blank lines. If the input does not contain
multiple paragraphs, non-empty lines are used as document units.

### 2. Tokenize and normalize

The tokenizer extracts alphanumeric terms and records each token's
position and character offsets. Terms are lowercased, stop words are
excluded from indexing, and the remaining terms are stemmed.

### 3. Build indexes

The search engine builds these in-memory structures:

  -----------------------------------------------------------------------
  Index / structure       Stored data             Use
  ----------------------- ----------------------- -----------------------
  Positional inverted     Stem → document ID →    Term retrieval and term
  index                   token positions         frequency

  Surface-form map        Stem → original         Associates normalized
                          lowercase word forms    stems with observed
                                                  forms

  K-gram index            Character bigram →      Wildcard and
                          vocabulary terms        spelling-correction
                                                  candidate generation

  Soundex index           Soundex code →          Phonetic matching
                          vocabulary terms        

  Document metadata       Document text and       Result display and
                          indexed-term counts     basic document
                                                  statistics
  -----------------------------------------------------------------------

### 4. Parse and evaluate the query

The query is tokenized and evaluated with a recursive-descent parser.
Boolean operations are implemented using Python set union, intersection,
and difference.

### 5. Rank and display results

Matched documents receive a TF-IDF-style score. Results are sorted by
descending score and limited to the first 50 matches. The GUI displays
document text, relevance values, search notes, and highlighted matching
terms.

The score contribution used for each query term is:

``` text
(1 + log10(term_frequency)) * log10(1 + document_count / document_frequency)
```

The contributions for the query terms are summed for each document.

------------------------------------------------------------------------

## Project Structure

The project is intentionally implemented as a single Python source file:

``` text
Text-Search-Engine/
├── search_engine.py
└── README.md
```

The source file is organized into these logical sections:

1.  Configuration and text-processing helpers
2.  Search engine and index construction
3.  Query tokenization, parsing, and retrieval
4.  Tkinter desktop application
5.  Main application entry point

------------------------------------------------------------------------

## Current Scope and Limitations

-   Only plain-text `.txt` input is supported by the file picker.
-   The application indexes one selected corpus at a time.
-   Indexes are stored in memory and are not persisted between runs.
-   The fallback stemmer is intentionally lightweight and is not a
    complete linguistic stemmer.
-   Spell correction is heuristic and may not find a suggestion for
    every typo.
-   Wildcard candidate generation may not return the expected expansion
    for every pattern.
-   A maximum of 50 matching documents is returned by the search method.
-   The application is designed as a compact educational project rather
    than a production-scale search service.

## Possible Future Improvements

-   Add automated unit tests for indexing, query parsing, phrase
    matching, and ranking.
-   Improve wildcard candidate generation and test prefix, suffix, and
    infix patterns.
-   Improve spelling correction and provide clearer correction
    suggestions.
-   Support additional document formats such as PDF and DOCX.
-   Add index export/import so a corpus does not need to be indexed
    every time.
-   Add filters, snippets, and more detailed result navigation.

------------------------------------------------------------------------

## Example Demonstration

For a quick demonstration, prepare a `.txt` file containing several
paragraphs on different topics. Load it into the application and try:

``` text
"machine learning"
cat AND NOT dog
(comp* OR information) AND NOT retrieval
soundex:smythe
informaton
```

The exact results depend on the words and paragraphs in the corpus you
load.

------------------------------------------------------------------------

## Learning Outcomes

This project provides practical experience with:

-   Regular expressions and text preprocessing
-   Dictionaries, sets, lists, and nested index structures
-   Inverted and positional indexing
-   Recursive-descent parsing
-   Boolean retrieval using set operations
-   K-gram indexing and wildcard matching
-   Soundex phonetic encoding
-   Levenshtein edit distance
-   TF-IDF-style relevance scoring
-   Building a desktop interface with Tkinter

------------------------------------------------------------------------

## License

No license has been specified in the submitted source. Add a license
file if you intend to publish or distribute the project for reuse.

## Author

Add your name, institution, course, and contact details here.
