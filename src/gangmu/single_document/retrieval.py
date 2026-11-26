"""Deterministic LaTeX-aware BM25 retrieval, never a dependency classifier.

Scores use Robertson IDF and BM25 term-frequency/length normalization. Common
mathematical vocabulary and generic symbols receive weak query weights; custom
TeX command case is preserved. The parser is intentionally lexical and bounded.
"""
from __future__ import annotations

from collections import Counter
import math
import re

VERSION = 'latex-bm25-v1'
PARAMETERS = {'k1': 1.2, 'b': 0.75, 'generic_word_weight': 0.15,
              'generic_symbol_weight': 0.05}
_STOP = set('a an the of on in to for and or is are we our by from with as at be '
            'this that its it these those has have then thus there here such '
            'definition theorem lemma proposition corollary proof statement '
            'expression notation introduction result let given consider using '
            'denote denotes write writes called defined named'.split())
_GENERIC_WORDS = set('module modules algebra algebras category categories set sets '
                    'complex complexes object objects map maps morphism morphisms '
                    'finite dimensional vector space exact sequence basic pair '
                    'property properties following corresponding'.split())
_GREEK = set('alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta '
             'iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau '
             'upsilon phi varphi chi psi omega'.split())
_GENERIC_COMMANDS = _GREEK | set('sum prod int iint oint cdot times otimes oplus '
    'odot circ star le ge leq geq neq ne eq in notin ni subset subseteq supset '
    'supseteq cup cap bigcup bigcap emptyset varnothing forall exists neg cong '
    'simeq sim approx equiv perp parallel mid vert Vert to gets rightarrow leftarrow '
    'leftrightarrow Rightarrow Leftarrow Leftrightarrow hookrightarrow mapsto '.split())
_STYLE = set('text textbf textit textsc emph operatorname mathrm mathbf mathcal '
    'mathbb mathsf mathit mathfrak mathscr boldsymbol bm rm bf it sf tt cal '
    'begin end label ref eqref autoref cref Cref vref cite citep citet '
    'frac dfrac tfrac overline underline widetilde widehat tilde hat bar vec dot '
    'ddot sqrt left right middle bigl bigr Bigl Bigr biggl biggr Biggl Biggr '
    'quad qquad hspace vspace displaystyle textstyle scriptstyle scriptscriptstyle '
    'dots ldots cdots vdots ddots item par smallskip medskip bigskip noindent '.split())
_COMMAND = re.compile(r'\\([A-Za-z@]+)')
_WORD = re.compile(r'[A-Za-z][A-Za-z-]{1,}')
_OPERATORS = re.compile(r'\\(?:operatorname\s*\*?|mathrm|text)\s*\{\s*([A-Za-z][A-Za-z-]+)\s*\}'
                        r'|\\rm\b\s*(?:\{\s*)?([A-Za-z][A-Za-z-]+)\b')
_HIDDEN_ARGUMENT = re.compile(r'\\(?:label|ref|eqref|autoref|[cC]ref|vref|cite[pt]?|begin|end)'
                              r'\s*(?:\[[^\]\n]*\]\s*)?\{[^{}\n]*\}')


def ignored_macros(structure_data) -> set[str]:
    """Declared layout/structure macros are not mathematical retrieval terms."""
    names = set()
    for entry in structure_data.get('definitions', []):
        if entry.get('definition_kind') != 'macro' or not entry.get('name'):
            continue
        body = entry.get('body') or ''
        if (entry.get('contains_structure') or
                re.search(r'\\(?:par|hspace|vspace|smallskip|medskip|bigskip|newpage|pagebreak)\b', body) or
                entry.get('argument_count', 0) > 0 and re.fullmatch(
                    r'\s*\\(?:textbf|textit|textsc|emph|text|mathbf|mathrm|mathsf|mathcal|mathbb|mathit|mathfrak|mathscr)'
                    r'\s*\{\s*#\d+\s*\}\s*', body)):
            names.add(entry['name'])
    return names


def token_spans(text: str, *, ignored=()) -> list[tuple[str, int, int]]:
    """Return normalized terms and exact original offsets, without expansion."""
    active = re.sub(r'(?<!\\)%[^\n]*', lambda match: ' ' * len(match[0]), text)
    active = _HIDDEN_ARGUMENT.sub(lambda match: ' ' * len(match[0]), active)
    occupied, terms = [], []
    for match in _OPERATORS.finditer(active):
        name = (match[1] or match[2]).lower()
        terms.append(('operator:' + name, match.start(), match.end()))
        occupied.append((match.start(), match.end()))
    for match in _COMMAND.finditer(active):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        name = match[1]
        occupied.append((match.start(), match.end()))
        if name in _STYLE or name in ignored:
            continue
        prefix = 'symbol:' if name.lower() in _GENERIC_COMMANDS else 'macro:'
        terms.append((prefix + name, match.start(), match.end()))
    for match in _WORD.finditer(active):
        if any(left <= match.start() < right for left, right in occupied):
            continue
        word = match[0].lower()
        if word not in _STOP:
            terms.append((word, match.start(), match.end()))
    return sorted(terms, key=lambda item: (item[1], item[2], item[0]))


def tokenize(text: str, *, ignored=()) -> list[str]:
    return [term for term, _, _ in token_spans(text, ignored=ignored)]


def term_weight(term: str) -> float:
    if term.startswith('symbol:'):
        return PARAMETERS['generic_symbol_weight']
    if term in _GENERIC_WORDS or term in {'operator:hom', 'operator:end', 'operator:mod'}:
        return PARAMETERS['generic_word_weight']
    return 1.0


class BM25Index:
    """Inverted index over token Counters with deterministic tie ordering."""

    def __init__(self, documents, *, k1=1.2, b=0.75):
        if not isinstance(k1, (int, float)) or isinstance(k1, bool) or not math.isfinite(k1) or k1 <= 0:
            raise ValueError('BM25 k1 must be finite and positive')
        if not isinstance(b, (int, float)) or isinstance(b, bool) or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError('BM25 b must be finite and in [0,1]')
        self.k1, self.b = float(k1), float(b)
        self.documents = [Counter(document) for document in documents]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.postings = {}
        for index, document in enumerate(self.documents):
            for term, frequency in document.items():
                if frequency > 0:
                    self.postings.setdefault(term, []).append((index, frequency))

    def idf(self, term):
        frequency = len(self.postings.get(term, ()))
        return math.log(1 + (len(self.documents) - frequency + .5) / (frequency + .5))

    def search(self, query, *, weights=None, eligible=None):
        terms = set(query)
        scores, shared = {}, {}
        for term in sorted(terms):
            weight = term_weight(term) * (weights.get(term, 1.0) if weights else 1.0)
            if weight <= 0:
                continue
            for index, frequency in self.postings.get(term, ()):
                if eligible is not None and index not in eligible:
                    continue
                normalization = (1 - self.b + self.b * self.lengths[index] / self.average_length
                                 if self.average_length else 1.0)
                component = (self.idf(term) * frequency * (self.k1 + 1) /
                             (frequency + self.k1 * normalization))
                scores[index] = scores.get(index, 0.0) + weight * component
                shared.setdefault(index, []).append(term)
        return [{'index': index, 'score': score,
                 'shared_terms': sorted(shared[index], key=lambda term: (-term_weight(term) * self.idf(term), term))}
                for index, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0]
