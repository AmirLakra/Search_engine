"""
TEXT SEARCH ENGINE
A complete single-file desktop search engine using Python's standard library.

Features:
- Paragraph-based document splitting
- Tokenization with original character offsets and positions
- Stop-word removal and lightweight stemming
- Positional inverted index
- K-gram index for wildcard expansion and spelling correction
- Boolean queries: AND, OR, NOT, parentheses (case-insensitive)
- Phrase queries: "machine learning"
- Wildcard queries: comp*, c?t, etc.
- Phonetic queries: soundex:smith or ~smith
- Automatic spell correction
- TF-IDF-style relevance ranking
- Tkinter GUI with highlighted matches

Run:
    python text_search_engine.py

Optional:
    pip install nltk
If NLTK is installed, its Porter stemmer is used. Otherwise a small
built-in suffix stemmer is used.
"""

import math
import re
import tkinter as tk
from collections import defaultdict
from tkinter import filedialog, messagebox, ttk


# ---------------------------------------------------------------------------
# 1. CONFIGURATION AND TEXT PROCESSING
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "hers",
    "him", "his", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "ours", "she", "so", "than",
    "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "to", "too", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "you", "your"
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
QUERY_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()]+')


try:
    from nltk.stem import PorterStemmer
    _PORTER = PorterStemmer()
except Exception:
    _PORTER = None


def stem(word):
    """Lowercase and stem a word; use NLTK Porter when available."""
    word = word.lower().strip("'")
    if not word:
        return word

    if _PORTER is not None:
        return _PORTER.stem(word)

    # Small dependency-free suffix stemmer. It is intentionally conservative.
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        if len(base) > 2 and base[-1:] == base[-2:-1]:
            base = base[:-1]
        return base
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if len(word) > 5 and word.endswith("ly"):
        return word[:-2]
    return word


def tokenize(text):
    """
    Return (original token, position, start offset, end offset).
    Positions count stop words too, which is important for phrase search.
    """
    for position, match in enumerate(TOKEN_RE.finditer(text)):
        yield match.group(), position, match.start(), match.end()


def soundex(word):
    """Classic four-character American Soundex code."""
    word = re.sub(r"[^A-Za-z]", "", word).upper()
    if not word:
        return ""

    codes = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        "L": "4",
        **dict.fromkeys("MN", "5"),
        "R": "6",
    }

    first = word[0]
    result = [first]
    previous = codes.get(first, "")

    for char in word[1:]:
        current = codes.get(char, "")
        if current and current != previous:
            result.append(current)
        previous = current
        if len(result) == 4:
            break

    return ("".join(result) + "000")[:4]


def edit_distance(a, b):
    """Levenshtein edit distance using dynamic programming."""
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb)
            ))
        previous = current
    return previous[-1]


def kgrams(word, k=2):
    """Return padded character k-grams."""
    padded = "$" + word.lower() + "$"
    if len(padded) < k:
        return {padded}
    return {padded[i:i + k] for i in range(len(padded) - k + 1)}


def split_documents(text):
    """Split a text file into paragraphs; fall back to non-empty lines."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
    return paragraphs


# ---------------------------------------------------------------------------
# 2. SEARCH ENGINE AND INDEX CONSTRUCTION
# ---------------------------------------------------------------------------

class SearchEngine:
    def __init__(self):
        self.documents = []
        self.inverted = defaultdict(lambda: defaultdict(list))
        self.surface_to_stem = defaultdict(set)
        self.kgram_index = defaultdict(set)
        self.soundex_index = defaultdict(set)
        self.doc_lengths = {}
        self.vocabulary = set()
        self.last_notes = []
        self.last_terms = set()

    def clear(self):
        self.__init__()

    def build(self, text):
        """Build all indexes from a complete corpus string."""
        self.clear()
        chunks = split_documents(text)

        for doc_id, content in enumerate(chunks):
            self.documents.append(content)
            terms_in_doc = 0

            for original, position, start, end in tokenize(content):
                lower = original.lower().strip("'")
                if not lower:
                    continue

                root = stem(lower)
                self.surface_to_stem[root].add(lower)

                if lower in STOP_WORDS:
                    continue

                self.inverted[root][doc_id].append(position)
                self.vocabulary.add(root)
                terms_in_doc += 1

            self.doc_lengths[doc_id] = terms_in_doc

        for word in self.vocabulary:
            for gram in kgrams(word):
                self.kgram_index[gram].add(word)
            self.soundex_index[soundex(word)].add(word)

        return len(self.documents), len(self.vocabulary)

    @property
    def document_count(self):
        return len(self.documents)

    def _postings(self, term):
        return set(self.inverted.get(stem(term), {}).keys())

    def _all_docs(self):
        return set(range(self.document_count))

    def _phrase(self, phrase):
        words = [m.group().lower().strip("'") for m in TOKEN_RE.finditer(phrase)]
        if not words:
            return set(), set()

        roots = [stem(word) for word in words]
        candidates = None

        for root in roots:
            if root in STOP_WORDS:
                continue
            docs = set(self.inverted.get(root, {}).keys())
            candidates = docs if candidates is None else candidates & docs

        if candidates is None:
            return set(), set()

        matches = set()
        for doc_id in candidates:
            position_sets = []
            for root in roots:
                if root in STOP_WORDS:
                    # Stop-word positions are not indexed, so use original text.
                    position_sets.append(None)
                else:
                    position_sets.append(set(self.inverted[root].get(doc_id, [])))

            # Verify the phrase against all token positions, including stop words.
            doc_tokens = list(tokenize(self.documents[doc_id]))
            normalized = [stem(tok.lower().strip("'")) for tok, _, _, _ in doc_tokens]
            for i in range(len(normalized) - len(roots) + 1):
                if normalized[i:i + len(roots)] == roots:
                    matches.add(doc_id)
                    break

        return matches, set(roots)

    def _wildcard_words(self, pattern):
        pattern = pattern.lower()
        # K-gram candidate generation, followed by exact glob-style matching.
        grams = set()
        pieces = re.split(r"[*?]+", pattern)
        for piece in pieces:
            if piece:
                grams |= kgrams(piece)
        if grams:
            candidate_sets = [self.kgram_index[g] for g in grams if g in self.kgram_index]
            candidates = set.intersection(*candidate_sets) if candidate_sets else set(self.vocabulary)
        else:
            candidates = set(self.vocabulary)

        regex = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
        matcher = re.compile(regex)
        return {word for word in candidates if matcher.match(word)}

    def _spell_correct(self, word):
        """Suggest a vocabulary term using bigram overlap and edit distance."""
        word = stem(word.lower())
        grams = kgrams(word)
        candidates = set()
        for gram in grams:
            candidates.update(self.kgram_index.get(gram, set()))

        scored = []
        for candidate in candidates:
            other = kgrams(candidate)
            union = grams | other
            similarity = len(grams & other) / len(union) if union else 0
            if similarity < 0.20:
                continue

            distance = edit_distance(word, candidate)
            cap = 1 if len(word) <= 4 else 2
            if distance <= cap:
                frequency = sum(len(p) for p in self.inverted[candidate].values())
                scored.append((distance, -frequency, candidate))

        if not scored:
            return None
        scored.sort()
        return scored[0][2]

    def _term(self, raw):
        raw = raw.strip()
        if not raw:
            return set(), set()

        lower = raw.lower()

        if lower.startswith("soundex:"):
            query_word = lower.split(":", 1)[1]
            code = soundex(query_word)
            words = self.soundex_index.get(code, set())
            self.last_notes.append(
                f"Soundex {code} for '{query_word}': {', '.join(sorted(words)) or 'no matches'}"
            )
            docs = set()
            terms = set()
            for word in words:
                docs.update(self._postings(word))
                terms.add(word)
            return docs, terms

        if lower.startswith("~"):
            return self._term("soundex:" + lower[1:])

        if "*" in lower or "?" in lower:
            words = self._wildcard_words(lower)
            self.last_notes.append(
                f"Wildcard '{lower}' expanded to: {', '.join(sorted(words)) or 'no terms'}"
            )
            docs = set()
            for word in words:
                docs.update(self._postings(word))
            return docs, set(words)

        query_stem = stem(lower)
        if query_stem in STOP_WORDS:
            return set(), set()

        if query_stem in self.inverted:
            return self._postings(query_stem), {query_stem}

        correction = self._spell_correct(lower)
        if correction:
            self.last_notes.append(f"Spell-corrected '{lower}' → '{correction}'")
            return self._postings(correction), {correction}

        return set(), {query_stem}

    # -----------------------------------------------------------------------
    # 3. QUERY PARSER
    # Grammar: OR -> AND -> NOT -> atom
    # Adjacent terms are treated as implicit AND.
    # -----------------------------------------------------------------------

    def _query_tokens(self, query):
        tokens = QUERY_RE.findall(query)
        normalized = []
        for token in tokens:
            if token.startswith('"') and token.endswith('"'):
                normalized.append(("PHRASE", token[1:-1]))
            elif token == "(":
                normalized.append(("LPAREN", token))
            elif token == ")":
                normalized.append(("RPAREN", token))
            elif token.upper() in {"AND", "OR", "NOT"}:
                normalized.append((token.upper(), token.upper()))
            else:
                normalized.append(("TERM", token))
        return normalized

    def search(self, query):
        self.last_notes = []
        self.last_terms = set()
        if not query.strip() or not self.documents:
            return []

        self.tokens = self._query_tokens(query)
        self.token_index = 0

        try:
            matched = self._parse_or()
            if self.token_index != len(self.tokens):
                raise ValueError("Unexpected query term. Check your operators and parentheses.")
        except (ValueError, IndexError) as exc:
            raise ValueError(str(exc))

        results = []
        for doc_id in matched:
            score = self._score(doc_id, self.last_terms)
            results.append((doc_id, score))

        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:50]

    def _peek(self):
        if self.token_index >= len(self.tokens):
            return None
        return self.tokens[self.token_index][0]

    def _take(self):
        token = self.tokens[self.token_index]
        self.token_index += 1
        return token

    def _parse_or(self):
        result = self._parse_and()
        while self._peek() == "OR":
            self._take()
            result |= self._parse_and()
        return result

    def _parse_and(self):
        result = self._parse_not()
        while True:
            kind = self._peek()
            if kind == "AND":
                self._take()
                result &= self._parse_not()
            elif kind in {"TERM", "PHRASE", "LPAREN", "NOT"}:
                # Implicit AND between adjacent query atoms.
                result &= self._parse_not()
            else:
                break
        return result

    def _parse_not(self):
        if self._peek() == "NOT":
            self._take()
            return self._all_docs() - self._parse_not()
        return self._parse_atom()

    def _parse_atom(self):
        kind = self._peek()
        if kind is None:
            raise ValueError("Incomplete query.")

        if kind == "LPAREN":
            self._take()
            result = self._parse_or()
            if self._peek() != "RPAREN":
                raise ValueError("Missing closing parenthesis.")
            self._take()
            return result

        if kind == "RPAREN":
            raise ValueError("Unexpected closing parenthesis.")

        token_kind, value = self._take()
        if token_kind == "PHRASE":
            docs, terms = self._phrase(value)
            self.last_terms.update(terms)
            return docs

        docs, terms = self._term(value)
        self.last_terms.update(terms)
        return docs

    def _score(self, doc_id, query_terms):
        """TF-IDF-style score for the matched query terms."""
        total = 0.0
        n_docs = max(self.document_count, 1)

        for term in query_terms:
            positions = self.inverted.get(term, {}).get(doc_id, [])
            tf = len(positions)
            df = len(self.inverted.get(term, {}))
            if tf and df:
                total += (1 + math.log10(tf)) * math.log10(1 + n_docs / df)

        return total

    def matching_offsets(self, doc_id, query_terms):
        """Find character ranges for query terms in a result document."""
        content = self.documents[doc_id]
        ranges = []

        for match in TOKEN_RE.finditer(content):
            word = match.group().lower().strip("'")
            root = stem(word)
            if root in query_terms or word in query_terms:
                ranges.append((match.start(), match.end()))

        return ranges


# ---------------------------------------------------------------------------
# 4. TKINTER DESKTOP APPLICATION
# ---------------------------------------------------------------------------

class SearchApp:
    """Minimal, AI-search-inspired interface matching the supplied reference."""

    BG = "#F8F8FB"
    INK = "#17191F"
    MUTED = "#747982"
    PANEL = "#E9ECF1"
    ACCENT = "#7898D4"
    WHITE = "#FFFFFF"

    def __init__(self, root):
        self.root = root
        self.root.title("Text Search Engine")
        self.root.geometry("1050x780")
        self.root.minsize(720, 620)
        self.root.configure(bg=self.BG)

        self.engine = SearchEngine()
        self.file_name = None

        self._build_ui()

    def _build_ui(self):
        self.page = tk.Frame(self.root, bg=self.BG)
        self.page.pack(fill="both", expand=True)

        # Header
        self.header = tk.Frame(self.page, bg=self.BG)
        self.header.pack(fill="x", pady=(62, 18))

        tk.Label(
            self.header, text="Text Search Engine",
            bg=self.BG, fg=self.INK,
            font=("Poppins", 30, "bold")
        ).pack()

        tk.Label(
            self.header,
            text="Ask detailed questions and find information across your documents",
            bg=self.BG, fg="#62666E",
            font=("Poppins", 13)
        ).pack(pady=(5, 0))

        # Search box canvas gives the input area a rounded, soft surface.
        self.search_canvas = tk.Canvas(
            self.page, height=150, bg=self.BG,
            highlightthickness=0, bd=0
        )
        self.search_canvas.pack(fill="x", padx=42)
        self.search_canvas.bind("<Configure>", self._draw_search_panel)

        self.query_var = tk.StringVar()
        self.query_entry = tk.Entry(
            self.search_canvas, textvariable=self.query_var,
            bd=0, relief="flat", bg=self.PANEL, fg="#333943",
            insertbackground=self.INK, font=("Poppins", 13)
        )
        self.query_entry.bind("<Return>", lambda _event: self.perform_search())

        self.load_btn = tk.Button(
            self.search_canvas, text="＋  Load text file",
            command=self.load_file, bd=0, relief="flat",
            bg=self.PANEL, fg="#242832", activebackground="#DDE2EA",
            font=("Poppins", 10), cursor="hand2"
        )
        self.clear_btn = tk.Button(
            self.search_canvas, text="Clear",
            command=self.clear_search, bd=0, relief="flat",
            bg=self.PANEL, fg="#555B66", activebackground="#DDE2EA",
            font=("Poppins", 10), cursor="hand2"
        )
        self.search_btn = tk.Button(
            self.search_canvas, text="➜",
            command=self.perform_search, bd=0, relief="flat",
            bg=self.ACCENT, fg=self.WHITE, activebackground="#6687C6",
            activeforeground=self.WHITE, font=("Poppins", 17, "bold"),
            cursor="hand2"
        )

        # Corpus status line
        self.status_var = tk.StringVar(value="No document loaded")
        self.status_label = tk.Label(
            self.page, textvariable=self.status_var,
            bg=self.BG, fg="#77808C", font=("Poppins", 9)
        )
        self.status_label.pack(pady=(4, 18))

        # Suggestions mimic the understated suggestion rows in the reference.
        self.suggestions = tk.Frame(self.page, bg=self.BG)
        self.suggestions.pack(fill="x", padx=78, pady=(0, 8))

        tk.Label(
            self.suggestions, text="SUGGESTED SEARCHES",
            bg=self.BG, fg="#9298A1",
            font=("Poppins", 8, "bold")
        ).pack(anchor="w", pady=(0, 8))

        examples = [
            ("⌕", 'Search for "machine learning"'),
            ("⌕", "Find documents containing cat AND NOT dog"),
            ("⌕", "Try wildcard search: comp*"),
            ("⌕", "Correct a spelling: informaton"),
        ]
        for symbol, label in examples:
            row = tk.Frame(self.suggestions, bg=self.BG)
            row.pack(fill="x", pady=4)
            tk.Label(
                row, text=symbol, bg=self.BG, fg="#9AA0AA",
                font=("Poppins", 13)
            ).pack(side="left", padx=(0, 12))
            link = tk.Label(
                row, text=label, bg=self.BG, fg="#858B94",
                font=("Poppins", 10), cursor="hand2"
            )
            link.pack(side="left")
            link.bind("<Button-1>", lambda _e, q=label: self._use_suggestion(q))

        # Results panel is kept below suggestions and becomes useful after search.
        self.results_shell = tk.Frame(
            self.page, bg=self.WHITE,
            highlightthickness=1, highlightbackground="#E8EAF0"
        )
        self.results_shell.pack(
            fill="both", expand=True, padx=34, pady=(2, 14)
        )

        self.results_heading = tk.Label(
            self.results_shell, text="RESULTS",
            bg=self.WHITE, fg="#777F8B",
            font=("Poppins", 10, "bold"), anchor="w"
        )
        self.results_heading.pack(fill="x", padx=20, pady=(15, 6))

        text_wrap = tk.Frame(self.results_shell, bg=self.WHITE)
        text_wrap.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.results = tk.Text(
            text_wrap, wrap="word", font=("Poppins", 11),
            bg=self.WHITE, fg="#303640", relief="flat",
            padx=8, pady=8, spacing1=2, spacing3=5,
            state="disabled", height=8
        )
        scrollbar = tk.Scrollbar(
            text_wrap, orient="vertical", command=self.results.yview
        )
        self.results.configure(yscrollcommand=scrollbar.set)
        self.results.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.results.tag_configure(
            "heading", foreground="#4D68B5",
            font=("Poppins", 10, "bold")
        )
        self.results.tag_configure(
            "meta", foreground="#858B94",
            font=("Poppins", 9, "italic")
        )
        self.results.tag_configure(
            "match", background="#FFF0A6", foreground="#202020"
        )
        self.results.tag_configure(
            "note", foreground="#16805D",
            font=("Poppins", 9, "italic")
        )
        self.results.tag_configure(
            "error", foreground="#C0392B",
            font=("Poppins", 10, "bold")
        )

        self._write(
            "Your search results will appear here.\n"
            "Load a .txt file, then enter a question or search expression above."
        )

        self.query_entry.focus_set()

    def _rounded_rectangle(self, canvas, x1, y1, x2, y2, radius=26, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1,
            x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius,
            x1, y1 + radius, x1, y1
        ]
        return canvas.create_polygon(
            points, smooth=True, splinesteps=24, **kwargs
        )

    def _draw_search_panel(self, event=None):
        canvas = self.search_canvas
        width = max(canvas.winfo_width(), 400)
        canvas.delete("panel")
        self._rounded_rectangle(
            canvas, 2, 8, width - 2, 142, radius=28,
            fill=self.PANEL, outline="#DCE2EB", width=1, tags="panel"
        )
        canvas.create_line(
            30, 9, width - 30, 9,
            fill="#D9E4F5", width=2, tags="panel"
        )

        # Place controls over the drawn rounded search surface.
        canvas.create_window(
            30, 48, window=self.query_entry,
            anchor="w", width=max(180, width - 190), height=34
        )
        canvas.create_window(
            25, 111, window=self.load_btn,
            anchor="w", width=130, height=30
        )
        canvas.create_window(
            157, 111, window=self.clear_btn,
            anchor="w", width=58, height=30
        )
        canvas.create_window(
            width - 42, 111, window=self.search_btn,
            anchor="center", width=42, height=42
        )

    def _write(self, text, tag=None):
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("end", text, tag)
        self.results.configure(state="disabled")

    def _use_suggestion(self, label):
        if 'machine learning' in label:
            query = '"machine learning"'
        elif 'cat AND NOT dog' in label:
            query = 'cat AND NOT dog'
        elif 'comp*' in label:
            query = 'comp*'
        else:
            query = 'informaton'
        self.query_var.set(query)
        self.perform_search()

    def clear_search(self):
        self.query_var.set("")
        self._write(
            "Your search results will appear here.\n"
            "Load a .txt file, then enter a question or search expression above."
        )
        self.query_entry.focus_set()

    def load_file(self):
        path = filedialog.askopenfilename(
            title="Choose a text corpus",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8-sig") as file:
                content = file.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="latin-1") as file:
                    content = file.read()
            except OSError as exc:
                messagebox.showerror("File error", str(exc))
                return
        except OSError as exc:
            messagebox.showerror("File error", str(exc))
            return

        docs, terms = self.engine.build(content)
        self.file_name = path.replace("\\", "/").split("/")[-1]
        self.status_var.set(
            f"{self.file_name}  ·  {docs} documents  ·  {terms} unique terms indexed"
        )
        self._write(
            f"Document loaded successfully\n\n"
            f"{self.file_name}\n{docs} documents indexed · {terms} unique terms\n\n"
            "You can now search the loaded text."
        )

    def perform_search(self):
        query = self.query_var.get().strip()

        if not self.engine.documents:
            messagebox.showinfo("No document loaded", "Please load a .txt file first.")
            return
        if not query:
            messagebox.showinfo("Empty search", "Please enter a search query.")
            return

        try:
            hits = self.engine.search(query)
        except ValueError as exc:
            self._write(f"Query error: {exc}", "error")
            return

        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("end", f"Search: {query}\n", "meta")

        for note in self.engine.last_notes:
            self.results.insert("end", f"• {note}\n", "note")

        self.results.insert(
            "end", f"\n{len(hits)} matching document(s)\n\n", "heading"
        )

        if not hits:
            self.results.insert(
                "end", "No matching documents. Try another search."
            )
        else:
            for rank, (doc_id, score) in enumerate(hits, 1):
                self.results.insert(
                    "end", f"Document {doc_id + 1}   ",
                    "heading"
                )
                self.results.insert(
                    "end", f"Relevance {score:.3f} · Result {rank}\n",
                    "meta"
                )

                content = self.engine.documents[doc_id]
                start_index = self.results.index("end")
                self.results.insert("end", content + "\n\n")

                terms = self.engine.last_terms
                for match in TOKEN_RE.finditer(content):
                    word = match.group().lower().strip("'")
                    if stem(word) in terms or word in terms:
                        left = f"{start_index}+{match.start()}c"
                        right = f"{start_index}+{match.end()}c"
                        self.results.tag_add("match", left, right)

        self.results.configure(state="disabled")
        self.results.see("1.0")


# ---------------------------------------------------------------------------
# 5. MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    SearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
