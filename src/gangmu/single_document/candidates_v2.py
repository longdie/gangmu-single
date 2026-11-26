"""Historical workflow-v2 retrieval, preserved for immutable cache replay.

Lexical matches propose review candidates; they never create accepted edges.
Commands are case-sensitive, operators match equivalent TeX spellings, and
context-only matches remain explicitly marked outside the target node body.
"""
from __future__ import annotations
import re

_STOP = set("a an the of on in to for and or is are we our by from with as at be this that its it "
            "definition theorem lemma proposition corollary proof statement expression notation "
            "module modules algebra algebras category categories set sets complex complexes "
            "basic finite dimensional vector space let given exact sequence bijection "
            "introduction result mutation triangle object objects assumption assumptions "
            "consider using denote denotes write writes add".split())
# Exclude syntax/formatting and built-in generic symbols, not short custom macros.
_SYNTAX = set("begin end label ref eqref autoref cref vref cite citep citet text textbf textit textsc emph "
              "operatorname mathrm mathbf mathcal mathbb mathsf mathit mathfrak mathscr boldsymbol bm "
              "rm bf it sf tt cal newcommand renewcommand providecommand def let DeclareMathOperator "
              "DeclareRobustCommand ensuremath mbox hbox fbox par smallskip medskip bigskip noindent "
              "item section subsection subsubsection paragraph subparagraph chapter title caption "
              "quad qquad hspace vspace small big tiny large Large LARGE footnotesize scriptsize "
              "left right middle bigl bigr Bigl Bigr biggl biggr Biggl Biggr frac dfrac tfrac "
              "overline underline widetilde widehat tilde hat bar vec dot ddot sqrt phantom "
              "hphantom vphantom displaystyle textstyle scriptstyle scriptscriptstyle "
              "dots ldots cdots vdots ddots longrightarrow longleftarrow longleftrightarrow "
              "rightarrow leftarrow leftrightarrow hookrightarrow hookleftarrow mapsto longmapsto "
              "to gets Rightarrow Leftarrow Leftrightarrow imply implies iff "
              "sum prod int iint oint limits nolimits cdot times otimes oplus odot circ star "
              "le ge leq geq neq ne eq in notin ni subset subseteq supset supseteq "
              "cup cap union bigcup bigcap emptyset varnothing forall exists neg "
              "cong simeq sim approx equiv perp parallel mid vert Vert lvert rvert "
              "hom end ker coker im dim det rank log ln sin cos tan min max inf sup "
              "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa "
              "lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega".split())
_SYNTAX_LOWER = {name.lower() for name in _SYNTAX}
_GREEK_LOWER = set("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa "
                   "lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega".split())
_CALL_OPEN = r'\s*(?:\\(?:left|bigl|Bigl|biggl|Biggl)\s*)?[\(\[]'
_COMMON_OPERATORS = {'hom', 'end', 'ext', 'tor', 'ker', 'coker', 'im', 'dim', 'det', 'rank',
                     'log', 'ln', 'sin', 'cos', 'tan', 'min', 'max', 'inf', 'sup', 'mod'}
_OPERATOR_SOURCE = re.compile(r"\\(?:operatorname\s*\*?|mathrm|text)\s*\{\s*([A-Za-z][A-Za-z-]+)\s*\}"
                              r"|\\rm\b\s*(?:\{\s*)?([A-Za-z][A-Za-z-]+)\b")


def _active_text(text):
    # Preserve character offsets while removing ordinary comments. This is a
    # lexical filter, not a substitute for E1's bounded TeX interpretation.
    return re.sub(r"(?<!\\)%[^\n]*", lambda m: ' ' * len(m.group()), text)


def _operator_pattern(name):
    value = re.escape(name)
    return (r"\\(?:operatorname\s*\*?|mathrm|text)\s*\{\s*" + value + r"\s*\}"
            r"|\\rm\b\s*(?:\{\s*)?" + value + r"(?![A-Za-z])"
            r"|\\" + value + r"(?![A-Za-z])")


def _features(node, project):
    text = _active_text('\n'.join(project.resolve_span(s) for s in node['spans']))
    values = {}

    def add(kind, value, pattern, flags=0, priority=10, **metadata):
        values[(kind, value)] = {'kind': kind, 'value': value, 'pattern': re.compile(pattern, flags),
                                 'priority': priority, **metadata}

    label = re.sub(r"\b(?:Definition|Theorem|Lemma|Proposition|Corollary|Proof|Expression|Notation)\b|\d+(?:\.\d+)*",
                   ' ', node['label'], flags=re.I)
    phrases = re.findall(r"\\(?:emph|textbf|textit)\s*\{([^{}]{3,})\}", text)
    words = re.findall(r'[A-Za-z][A-Za-z-]{2,}', label)
    phrases.extend(w for w in words if w.lower() not in _STOP)
    phrases.extend(' '.join(words[i:i+2]) for i in range(len(words)-1)
                   if any(w.lower() not in _STOP for w in words[i:i+2]))
    for phrase in phrases:
        value = phrase.strip()
        if len(value) >= 3 and value.lower() not in _STOP:
            pattern = re.escape(value).replace(r'\ ', r'\s+')
            add('phrase', value, r'(?<![A-Za-z])' + pattern + r'(?![A-Za-z])', re.I, 10)
    declarations = {entry.get('name'): entry for entry in project.structure_data.get('definitions', [])
                    if entry.get('definition_kind') == 'macro' and entry.get('name')}
    for match in re.finditer(r'\\([A-Za-z]+)(?![A-Za-z])', text):
        name = match.group(1)
        value = '\\' + name
        if name == 'perp':
            # Orthogonality constructions are often explicitly defined with
            # this built-in symbol. Retain a weak cue, never an accepted edge.
            add('math_symbol', value, re.escape(value) + r'(?![A-Za-z])', priority=5)
            continue
        if name.lower() in _GREEK_LOWER:
            pattern = re.escape(value) + r'(?![A-Za-z])' + _CALL_OPEN
            if re.match(_CALL_OPEN, text[match.end():]):
                # A bare Greek parameter is generic. Explicit application is
                # useful for named constructions, with exact TeX command case.
                add('math_function', value + '(...)', pattern, priority=8)
            continue
        if name.lower() in _SYNTAX_LOWER:
            continue
        declaration = declarations.get(name)
        if declaration and (declaration.get('contains_structure') or
                            re.search(r'\\(?:par|hspace|vspace|smallskip|medskip|bigskip|newpage|pagebreak)\b',
                                      declaration.get('body') or '') or
                            (declaration.get('argument_count', 0) > 0 and re.fullmatch(
                                r'\s*\\(?:textbf|textit|textsc|emph|text|mathbf|mathrm|mathsf|mathcal|mathbb|mathit|mathfrak|mathscr)\s*\{\s*#\d+\s*\}\s*',
                                declaration.get('body') or ''))):
            continue
        # A custom macro is a retrieval symbol; a TeX definition is no semantic
        # certificate. Every finite surviving macro is kept, without top-k.
        add('macro', value, re.escape(value) + r'(?![A-Za-z])', priority=20,
            syntax_definition_known=declaration is not None)
    for match in _OPERATOR_SOURCE.finditer(text):
        name = match.group(1) or match.group(2)
        if name.lower() in _COMMON_OPERATORS or name.lower() in _STOP:
            # add is deliberately a plain-language stop word but a useful
            # mathematical operator when it occurs in an operator spelling.
            if name.lower() != 'add':
                continue
        add('operator', name, _operator_pattern(name), priority=25)
    return sorted(values.values(), key=lambda value: (-value['priority'], -len(value['value']),
                                                      value['kind'], value['value']))


def _terms(node, project):
    """Compatibility inspection API: all finite retained terms, no hidden cap."""
    return [feature['value'] for feature in _features(node, project)]


def _regions(target, draft):
    regions, seen = [], set()

    def add(span, scope, context=None):
        key = (span['file'], span['char_start'], span['char_end'], span.get('occurrence'))
        if key not in seen:
            seen.add(key)
            regions.append({'span': span, 'scope': scope, 'context': context})
    for span in target['spans']:
        add(span, 'node')
    contexts = draft.data.get('contexts', {})
    context_id, visited = target.get('context'), set()
    while context_id and context_id not in visited:
        visited.add(context_id)
        context = contexts.get(context_id)
        if not context:
            break
        for span in context.get('spans', []):
            add(span, 'context', context_id)
        context_id = context.get('parent')
    return regions


def discover(project, draft, node_id=None, offset=0, limit=50):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('invalid relation candidate pagination')
    nodes = [n for n in draft.data['nodes'].values() if n['judgment'] == 'accepted']
    by_id = {n['id']: n for n in nodes}
    if node_id is not None and node_id not in by_id:
        raise ValueError('candidate target must be an accepted registered node')
    profiles = {}
    for source in nodes:
        if source['category'] in ('definition', 'expression'):
            profiles[source['id']] = _features(source, project)
        else:
            patterns = re.findall(r'(?:Theorem|Lemma|Proposition|Corollary)\s+\d+(?:\.\d+)*',
                                  source['label'], flags=re.I)
            if source['category'] == 'proof' and re.search(r'\b(?:Lemma|Theorem|Proposition|Corollary)\b', source['label'], re.I):
                patterns = [source['label']]
            profiles[source['id']] = [{'kind': 'name', 'value': name, 'priority': 10,
                'pattern': re.compile(r'(?<![A-Za-z])' + re.escape(name).replace(r'\ ', r'\s+') + r'(?![A-Za-z])', re.I)}
                for name in patterns]
    suggestions = {}
    explicit_pages, context_regions = 0, 0

    def add(source, target, use, region, method, score, feature=None):
        if source['id'] == target['id'] or target['category'] != 'proof' and source['category'] == 'proof':
            return
        kind = 'proof_dependency' if target['category'] == 'proof' else 'statement_dependency'
        key = source['id'], target['id'], kind
        if key not in suggestions or (region['scope'] == 'node', score) > (
                suggestions[key]['use_scope'] == 'node', suggestions[key]['score']):
            suggestions[key] = {'source': source['id'], 'target': target['id'], 'type': kind,
                'source_label': source['label'], 'target_label': target['label'], 'method': method,
                'score': score, 'term': feature['value'] if feature else None,
                'term_kind': feature['kind'] if feature else None, 'use_span': use,
                'use_scope': region['scope'], 'context_id': region['context'],
                'distance': project.distance(use, source['spans'][0])}
    for target in nodes:
        if node_id is not None and target['id'] != node_id:
            continue
        for region in _regions(target, draft):
            context_regions += region['scope'] == 'context'
            span, scope = region['span'], region['scope']
            text = _active_text(project.resolve_span(span))
            cursor = 0
            while True:
                page = project.dependency_candidates_legacy([span], offset=cursor, limit=100)
                explicit_pages += 1
                for candidate in page['explicit']:
                    position = candidate['label_span']
                    for source in nodes:
                        if any(s['file'] == position['file'] and s['char_start'] <= position['char_start'] < s['char_end']
                               for s in source['spans']):
                            add(source, target, candidate['use_span'], region,
                                'explicit_label' if scope == 'node' else 'context_explicit_label', 100 if scope == 'node' else 75)
                if page['next_offset'] is None:
                    break
                cursor = page['next_offset']
            for source in nodes:
                if source['id'] == target['id']:
                    continue
                for feature in profiles[source['id']]:
                    match = feature['pattern'].search(text)
                    if match:
                        use = {'file': span['file'], 'char_start': span['char_start'] + match.start(),
                               'char_end': span['char_start'] + match.end()}
                        if span.get('occurrence') is not None:
                            use['occurrence'] = span['occurrence']
                        score = 10 + min(len(feature['value']), 40) + feature['priority']
                        if scope == 'context':
                            score = max(1, score - 15)
                        add(source, target, use, region, 'whole_inventory_term' if scope == 'node'
                            else 'whole_inventory_context_term', score, feature)
    records = sorted(suggestions.values(), key=lambda x: (-x['score'], x['target'], x['source']))
    selected = records[offset:offset+limit]
    return {'total': len(records), 'records': selected,
        'next_offset': offset + len(selected) if offset + len(selected) < len(records) else None,
        'inventory_size': len(by_id), 'term_features_total': sum(map(len, profiles.values())),
        'term_features_by_source': {key: len(value) for key, value in sorted(profiles.items())},
        'term_feature_cap': None, 'context_source_regions': context_regions, 'explicit_label_pages': explicit_pages,
        'search_scope': 'all accepted nodes, all finite surviving source terms, all explicit pages, declared context ancestry, all distances including forward',
        'notice': 'Candidates are not dependencies. Context use_span is setup source, not target-node text or automatic edge evidence. Verify actual use and expand lexical misses.'}
