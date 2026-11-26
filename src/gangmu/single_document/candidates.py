"""Finite whole-inventory retrieval from original nodes and declared contexts.

Lexical matches propose review candidates; they never create accepted edges.
Commands are case-sensitive, operators match equivalent TeX spellings, and
context-only matches remain explicitly marked outside the target node body.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
import copy
import re

from gangmu.representation.model import digest
from gangmu.single_document.retrieval import BM25Index, PARAMETERS as BM25_PARAMETERS, ignored_macros, term_weight, token_spans, tokenize

RETRIEVAL_VERSION = 'source-bound-union-v1'
RETRIEVAL_PARAMETERS = {'bm25_top_k_per_target': 6, 'local_each_direction': 2,
                        'context_query_weight': 0.25, 'context_document_weight': 0.25,
                        'bm25_relative_score_floor': 0.35, 'max_query_terms': 96,
                        'cache_entries': 2, **BM25_PARAMETERS}

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
              "overline underline widetilde widehat tilde hat bar vec dot ddot sqrt phantom buildrel "
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
# Bounded literal scripts, including a nested subscript such as P_{X}.  This
# recognizes a callable spelling, not its meaning or a general TeX expansion.
_SCRIPT_CONTENT = r'(?:[^{}\r\n]|\{[^{}\r\n]{0,64}\}){1,128}'
_SCRIPT_ATOM = r'(?:\{' + _SCRIPT_CONTENT + r'\}|\\[A-Za-z]{1,32}|[A-Za-z0-9+\-])'
_CALL_TAIL = r'(?:\s*[_^]\s*' + _SCRIPT_ATOM + r'){0,2}' + _CALL_OPEN
_EVIDENCE_LIMIT = 6
_NAMING_CUE = re.compile(r'\b(?:called|named)\s+(?:(?:the|a|an)\s+)?', re.I)
_DECLARATION_CUE = re.compile(r'\b(?:called|named|defined|denoted)\b|:=|=:', re.I)
_PREFIX_DECLARATION = re.compile(r'\b(?:we\s+)?(?:define|denote|introduce)\b', re.I)
_NAME_STOP = (_STOP - {'mutation'}) | {'pair', 'map', 'functor', 'property', 'construction'}
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
            pattern = re.escape(value) + r'(?![A-Za-z])' + _CALL_TAIL
            if re.match(_CALL_TAIL, text[match.end():]):
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


def _named_constructions(text):
    """Read short literal naming clauses; never infer a name from a label.

    The source must supply a naming cue.  Supported italic/math wrappers and
    plain words are only lexical aids; unbalanced or complicated TeX is left
    for the Agent's full-source reading rather than guessed here.
    """
    for cue in _NAMING_CUE.finditer(text):
        tail = text[cue.end():cue.end() + 160].lstrip()
        wrapper = re.match(r'\\(?:emph|textit|textbf|mathit)\s*', tail)
        if wrapper:
            tail = tail[wrapper.end():]
        if tail.startswith('{'):
            depth, end = 0, None
            for position, char in enumerate(tail):
                depth += (char == '{') - (char == '}')
                if depth == 0:
                    end = position; break
            if end is None:
                continue
            name = tail[1:end]
        elif tail.startswith('$'):
            end = tail.find('$', 1)
            if end < 0:
                continue
            name = tail[1:end]
        else:
            match = re.match(r'[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,4}', tail)
            if not match:
                continue
            name = re.split(r'\s+(?:of|with|for|by|in|on|if|and|which|that)\b',
                            match[0], maxsplit=1, flags=re.I)[0]
        name = re.sub(r'\\(?:it|rm|bf|sf)\b|[{}$]', ' ', name)
        name = ' '.join(name.split())
        if (3 <= len(name) <= 80 and re.fullmatch(r'[A-Za-z][A-Za-z -]*', name)
                and any(word.lower() not in _NAME_STOP for word in name.split())):
            yield name


def _construction_features(node, project):
    """A statement may introduce an operation while asserting a result.

    Keep only source-named constructions and the closest callable/operator
    at a literal declaration cue.  Unlike definition retrieval, arbitrary
    technical words or symbols used elsewhere in a statement are not profiles.
    These constrained cues remain advisory and can describe overloaded names.
    """
    text = _active_text('\n'.join(project.resolve_span(s) for s in node['spans']))
    values = {}
    for name in _named_constructions(text):
        words = name.split()
        terms = [('construction_name', name, 12)]
        if len(words) > 1 and words[-1].lower() not in _NAME_STOP:
            terms.append(('construction_head', words[-1], 3))
        for kind, value, priority in terms:
            pattern = re.escape(value).replace(r'\ ', r'\s+')
            values[(kind, value)] = {'kind': kind, 'value': value, 'priority': priority,
                'pattern': re.compile(r'(?<![A-Za-z])' + pattern + r'(?![A-Za-z])', re.I),
                'source_basis': 'literal_source_naming_clause', 'source_declared_name': name}
    cues = [(cue, False) for cue in _DECLARATION_CUE.finditer(text)]
    cues += [(cue, True) for cue in _PREFIX_DECLARATION.finditer(text)]
    if not cues:
        return []
    symbols = [(feature, match) for feature in _features(node, project)
               if feature['kind'] in ('math_function', 'operator', 'macro')
               for match in feature['pattern'].finditer(text)]
    for cue, following in cues:
        nearby = []
        for feature, match in symbols:
            gap = text[cue.end():match.start()] if following else text[match.end():cue.start()]
            ordered = match.start() >= cue.end() if following else match.end() <= cue.start()
            if ordered and len(gap) <= 160 and not re.search(r'[.!?](?:\s|$)', gap):
                # A callable is more specific than a category/operator appearing
                # inside its arguments.  Keep one nearby symbolic declaration
                # per cue instead of all the statement's input vocabulary.
                nearby.append((feature['kind'] == 'math_function', -len(gap), feature['priority'], feature))
        if nearby:
            feature = max(nearby, key=lambda item: item[:3])[-1]
            values[(feature['kind'], feature['value'])] = {**feature,
                'source_basis': 'symbol_at_literal_source_declaration',
                'priority': min(feature['priority'], 12)}
    return sorted(values.values(), key=lambda feature: (-feature['priority'], feature['kind'], feature['value']))


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


def _occurrence_compatible(left, right):
    return (left.get('occurrence') is None or right.get('occurrence') is None
            or left['occurrence'] == right['occurrence'])


def _intersects(left, right):
    return (left['file'] == right['file'] and _occurrence_compatible(left, right)
            and left['char_start'] < right['char_end']
            and right['char_start'] < left['char_end'])


def _reference_sources(nodes, target, *, occurrence_resolver=None, heading_owner_filter=None):
    """Return plausible mathematical owners, without certifying reference use.

    A label can follow a closing environment or belong to a nested expression.
    Exact label containment and independently located mathematical owners are
    distinct clues. Preserve alternative owners for the Agent to adjudicate;
    never invent an endpoint or choose the most convenient accepted node.
    """
    labels = target.get('label_spans', [])
    owners = target.get('math_owners', [])
    headings = target.get('heading_anchors', [])
    candidates = {}
    for node in nodes:
        direct = any(_intersects(span, label) for span in node['spans'] for label in labels)
        heading_matches = [heading for heading in headings
            if node['category'] in ('definition', 'statement')
            and (heading_owner_filter is None or heading_owner_filter(heading, node))
            and heading.get('source_spans') and all(any(
                span['file'] == anchor['file']
                and ((_occurrence_compatible(span, anchor) if occurrence_resolver is None else
                      occurrence_resolver(span) is not None and
                      occurrence_resolver(span) == occurrence_resolver(anchor)))
                and span['char_start'] <= anchor['char_start'] < anchor['char_end'] <= span['char_end']
                for span in node['spans']) for anchor in heading['source_spans'])]
        matched = []
        for owner in owners:
            spans = owner.get('source_spans', [])
            # A tiny nested display does not designate the entire surrounding
            # theorem; both-sided source overlap avoids that mechanical lift.
            left_total = sum(s['char_end'] - s['char_start'] for s in node['spans'])
            right_total = sum(s['char_end'] - s['char_start'] for s in spans)
            overlap = sum(max(0, min(a['char_end'], b['char_end']) - max(a['char_start'], b['char_start']))
                          for a in node['spans'] for b in spans if _intersects(a, b))
            if min(left_total, right_total) > 0 and overlap >= .5 * max(left_total, right_total):
                matched.append(owner)
        if direct or matched or heading_matches:
            basis = ('label_inside_node' if direct else 'mathematical_owner_overlap'
                     if matched else 'literal_printed_heading_inside_node_requires_review')
            candidates[node['id']] = (node, basis,
                [owner.get('event_id') for owner in matched] + [heading['item_id'] for heading in heading_matches])
    return list(candidates.values())



def _paired_proof_sources(nodes, relations, endpoint, *, occurrence_resolver=None, heading_owner_filter=None):
    """Current proof owners for this exact locator endpoint, never dependencies.

    A statement label can locate its separately stored argument. Only accepted
    same-project proof -> statement ``proves`` associations supply that bridge;
    an arbitrary incident relation or a different referenced key cannot do so.
    Multiple paired proofs remain alternatives for source reading, not a vote
    for whichever proof the Agent happens to want to use.
    """
    by_id = {node['id']: node for node in nodes}
    owners = {node['id']: (node, basis, events)
              for node, basis, events in _reference_sources(nodes, endpoint,
                  occurrence_resolver=occurrence_resolver, heading_owner_filter=heading_owner_filter)
              if node['category'] == 'statement'}
    records = []
    for relation in sorted(relations.values(), key=lambda value: value['id']):
        if relation['type'] != 'proves' or relation['judgment'] != 'accepted':
            continue
        proof = by_id.get(relation['source'])
        owner = owners.get(relation['target'])
        if proof is not None and proof['category'] == 'proof' and owner is not None:
            statement, basis, events = owner
            records.append((proof, statement, relation['id'], basis, events))
    return records


def _proof_reference_cue(project, reference):
    """A bounded literal proof-of hint immediately before this raw locator.

    This is retrieval advice only. Absence does not bar a manually identified
    proof dependency, and presence does not certify that the proof is reused.
    The end anchor prevents a cue for one locator leaking through to another.
    """
    use = reference.get('span', reference.get('source'))
    if not use or use['char_start'] <= 0:
        return None
    prefix = {**use, 'char_start': max(0, use['char_start'] - 160),
              'char_end': use['char_start']}
    text = _active_text(project.resolve_span(prefix))
    spacing = r'(?:\s|~|\\[,;:! ])'
    match = re.search(r'\bproofs?' + spacing + r'+of' + spacing +
        r'+(?:(?:the' + spacing + r'+)?(?:lemma|theorem|proposition|corollary|claim|statement|result)s?' +
        spacing + r'*)?$', text, flags=re.I)
    if match is None:
        return None
    return {**prefix, 'char_start': prefix['char_start'] + match.start()}


def _full_references(project):
    """Source-first ledger; no accepted-node or Context filters in its count."""
    if not hasattr(project, 'reference_ledger'):
        return None
    rows, offset, binding = [], 0, None
    while True:
        page = project.reference_ledger(offset=offset, limit=100)
        rows.extend(page['records'])
        binding = page.get('inventory_sha256')
        if page['next_offset'] is None:
            return rows, binding
        offset = page['next_offset']


def _source_order(project, node):
    """Reading order is a local-retrieval hint, never dependency direction."""
    positions = [item['reading_char_start'] + span['char_start'] - item['source']['char_start']
                 for span in node['spans'] for item in getattr(project, 'source_map', ())
                 if item['source']['file'] == span['file']
                 and item['source']['char_start'] <= span['char_start'] < item['source']['char_end']
                 and (span.get('occurrence') is None or item['occurrence'] == span['occurrence'])]
    first = node['spans'][0]
    return (not bool(positions), min(positions) if positions else 0,
            first['file'], first['char_start'], first.get('occurrence', ''), node['id'])


def _content_binding(project, draft, nodes):
    contexts, used = draft.data.get('contexts', {}), {}
    for node in nodes:
        identifier, visited = node.get('context'), set()
        while identifier and identifier not in visited:
            visited.add(identifier)
            context = contexts.get(identifier)
            used[identifier] = context
            if context is None:
                break
            identifier = context.get('parent')
    proves = [relation for relation in draft.data.get('relations', {}).values()
              if relation.get('type') == 'proves' and relation.get('judgment') == 'accepted']
    return digest({'version': RETRIEVAL_VERSION, 'parameters': RETRIEVAL_PARAMETERS,
                   'source': getattr(project, 'fingerprint', None),
                   'nodes': sorted(nodes, key=lambda node: node['id']), 'contexts': used,
                   'proves': sorted(proves, key=lambda relation: relation['id'])})


def _build_candidates(project, draft, nodes):
    by_id = {n['id']: n for n in nodes}
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
            if source['category'] == 'statement':
                profiles[source['id']].extend(_construction_features(source, project))
    suggestions = {}
    evidence = {}
    explicit_pages, context_regions = 0, 0
    ledger = _full_references(project)

    def add(source, target, use, region, method, score, feature=None, **metadata):
        if source['id'] == target['id'] or target['category'] != 'proof' and source['category'] == 'proof':
            return
        kind = 'proof_dependency' if target['category'] == 'proof' else 'statement_dependency'
        key = source['id'], target['id'], kind
        if feature and feature.get('source_basis'):
            metadata['source_feature_basis'] = feature['source_basis']
            if feature.get('source_declared_name'):
                metadata['source_declared_name'] = feature['source_declared_name']
        clue = {'method': method, 'score': score, 'term': feature['value'] if feature else None,
                'term_kind': feature['kind'] if feature else None, 'use_span': use,
                'use_scope': region['scope'], 'context_id': region['context'], **metadata}
        # Keep every explicitly located occurrence/key, including alternate
        # proof owners. Only the displayed use_evidence is bounded below.
        group = (region['scope'], clue['term_kind'] or method,
                 metadata.get('reference_id'), metadata.get('reference_key'),
                 metadata.get('reference_source_role'))
        groups = evidence.setdefault(key, {})
        if group not in groups or score > groups[group]['score']:
            groups[group] = clue
        if key not in suggestions or (region['scope'] == 'node', score) > (
                suggestions[key]['use_scope'] == 'node', suggestions[key]['score']):
            suggestions[key] = {'source': source['id'], 'target': target['id'], 'type': kind,
                'source_label': source['label'], 'target_label': target['label'], **clue,
                'distance': project.distance(use, source['spans'][0])}
    for target in nodes:
        for region in _regions(target, draft):
            context_regions += region['scope'] == 'context'
            span, scope = region['span'], region['scope']
            text = _active_text(project.resolve_span(span))
            if ledger is not None:
                for reference in ledger[0]:
                    use = reference.get('span', reference.get('source'))
                    if not use or not _intersects(span, use):
                        continue
                    for endpoint in reference.get('targets', []):
                        for source, basis, owner_ids in _reference_sources(nodes, endpoint):
                            add(source, target, use, region,
                                'explicit_label' if scope == 'node' else 'context_explicit_label',
                                100 if scope == 'node' else 75,
                                reference_id=reference['item_id'], reference_key=endpoint.get('key'),
                                endpoint_resolution=endpoint.get('resolution'),
                                target_ownership_basis=basis, mathematical_owner_event_ids=owner_ids)
                        cue = _proof_reference_cue(project, reference) if target['category'] == 'proof' else None
                        if cue is not None:
                            for proof, statement, proves_id, basis, owner_ids in _paired_proof_sources(
                                    nodes, draft.data['relations'], endpoint):
                                add(proof, target, use, region,
                                    'explicit_label' if scope == 'node' else 'context_explicit_label',
                                    100 if scope == 'node' else 75,
                                    reference_id=reference['item_id'], reference_key=endpoint.get('key'),
                                    endpoint_resolution=endpoint.get('resolution'),
                                    target_ownership_basis=basis, mathematical_owner_event_ids=owner_ids,
                                    reference_source_role='proof', referenced_statement_id=statement['id'],
                                    paired_proves_relation_id=proves_id, proof_cue_span=cue)
            else:
                # Compatibility for callers replaying a historical source
                # adapter. New production always has the complete ledger.
                cursor = 0
                while True:
                    page = project.dependency_candidates([span], offset=cursor, limit=100)
                    explicit_pages += 1
                    for candidate in page['explicit']:
                        position = candidate['label_span']
                        for source in nodes:
                            if any(_intersects(s, position) for s in source['spans']):
                                add(source, target, candidate['use_span'], region,
                                    'explicit_label' if scope == 'node' else 'context_explicit_label',
                                    100 if scope == 'node' else 75)
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

    # An economical union, not replacement of the complete explicit/semantic
    # clues above. Each target's own body is its query; context terms have a
    # weaker field weight. Statements and their proofs are never pooled.
    source_regions = {node['id']: _regions(node, draft) for node in nodes}
    bodies = {node['id']: '\n'.join(project.resolve_span(span) for span in node['spans'])
              for node in nodes}
    context_text = {node['id']: '\n'.join(project.resolve_span(region['span'])
                    for region in source_regions[node['id']] if region['scope'] == 'context')
                    for node in nodes}
    blocked_macros = ignored_macros(project.structure_data)
    corpus = [Counter(tokenize(bodies[node['id']], ignored=blocked_macros)) for node in nodes]
    index = BM25Index(corpus, k1=BM25_PARAMETERS['k1'], b=BM25_PARAMETERS['b'])
    context_index = BM25Index([Counter(tokenize(context_text[node['id']], ignored=blocked_macros)) for node in nodes],
                             k1=BM25_PARAMETERS['k1'], b=BM25_PARAMETERS['b'])
    ordered = sorted(nodes, key=lambda node: _source_order(project, node))
    for target in nodes:
        target_regions = source_regions[target['id']]
        matches = [(term, {**region['span'], 'char_start': region['span']['char_start'] + start,
                          'char_end': region['span']['char_start'] + end}, region)
                   for region in target_regions
                   for term, start, end in token_spans(project.resolve_span(region['span']), ignored=blocked_macros)]
        query_weights = {}
        for term, _, region in matches:
            query_weights[term] = max(query_weights.get(term, 0.0),
                1.0 if region['scope'] == 'node' else RETRIEVAL_PARAMETERS['context_query_weight'])
        query_terms = sorted(query_weights, key=lambda term: (
            -term_weight(term) * query_weights[term] * index.idf(term), term))[
                :RETRIEVAL_PARAMETERS['max_query_terms']]
        eligible = {position for position, source in enumerate(nodes)
                    if source['id'] != target['id'] and
                    (target['category'] == 'proof' or source['category'] != 'proof')}
        # Shared document-wide setup alone must not make every node relevant.
        # A source-body lexical match is required; source context boosts such
        # a match but cannot independently originate a BM25 candidate.
        context_hits = {hit['index']: hit for hit in context_index.search(
            query_terms, weights=query_weights, eligible=eligible)}
        retrieved = []
        for hit in index.search(query_terms, weights=query_weights, eligible=eligible):
            informative = [term for term in hit['shared_terms'] if term_weight(term) >= .5]
            if not informative:
                continue
            if len(informative) == 1:
                term = informative[0]
                if not term.startswith(('macro:', 'operator:')) and len(index.postings.get(term, ())) > max(2, len(nodes) * .25):
                    continue
            context_score = context_hits.get(hit['index'], {}).get('score', 0.0)
            retrieved.append({**hit, 'score': hit['score'] + RETRIEVAL_PARAMETERS['context_document_weight'] * context_score,
                              'body_score': hit['score'], 'context_score': context_score})
        retrieved.sort(key=lambda hit: (-hit['score'], hit['index']))
        if retrieved:
            floor = RETRIEVAL_PARAMETERS['bm25_relative_score_floor'] * retrieved[0]['score']
            retrieved = [hit for hit in retrieved if hit['score'] >= floor][
                :RETRIEVAL_PARAMETERS['bm25_top_k_per_target']]
        for rank, hit in enumerate(retrieved, 1):
            shared = set(hit['shared_terms'])
            locations = [value for value in matches if value[0] in shared]
            term, use, region = max(locations, key=lambda value: (
                value[2]['scope'] == 'node', term_weight(value[0]) * index.idf(value[0])))
            feature = {'kind': 'bm25_tokens', 'value': term}
            add(nodes[hit['index']], target, use, region, 'full_node_bm25',
                25 + min(20, hit['score'] * 2), feature,
                bm25_score=round(hit['score'], 6), bm25_rank=rank,
                bm25_body_score=round(hit['body_score'], 6), bm25_context_score=round(hit['context_score'], 6),
                shared_terms=hit['shared_terms'][:12], query_target_category=target['category'],
                position_kind='lexical_match', retrieval_only=True)
        pivot = ordered.index(target)
        for direction, candidates in (('before', reversed(ordered[:pivot])), ('after', ordered[pivot+1:])):
            count = 0
            for source in candidates:
                if target['category'] != 'proof' and source['category'] == 'proof':
                    continue
                count += 1
                region = next(region for region in target_regions if region['scope'] == 'node')
                add(source, target, region['span'], region, 'bidirectional_local_window',
                    8 - count, {'kind': 'local_window', 'value': direction},
                    local_direction=direction, local_rank=count,
                    position_kind='target_reading_envelope', retrieval_only=True)
                if count >= RETRIEVAL_PARAMETERS['local_each_direction']:
                    break
    for key, record in suggestions.items():
        clues = sorted(evidence[key].values(), key=lambda clue: (
            clue['use_scope'] == 'node', clue['score']), reverse=True)
        primary = next(clue for clue in clues if clue['use_span'] == record['use_span']
                       and clue['term_kind'] == record['term_kind'])
        # Reserve a distinct scope before filling the finite evidence display;
        # a strong body cue must not erase actual inherited setup (or vice versa).
        other_scope = next((clue for clue in clues if clue['use_scope'] != primary['use_scope']), None)
        selected_clues = [primary] + ([other_scope] if other_scope else [])
        for method in sorted({clue['method'] for clue in clues}):
            clue = next(clue for clue in clues if clue['method'] == method)
            if clue not in selected_clues:
                selected_clues.append(clue)
        selected_clues.extend(clue for clue in clues if clue not in selected_clues)
        record['use_evidence'] = selected_clues[:_EVIDENCE_LIMIT]
        record['retrieval_channels'] = sorted({clue['method'] for clue in clues})
        record['explicit_references'] = [clue for clue in clues if clue.get('reference_id') is not None]
    records = sorted(suggestions.values(), key=lambda x: (-x['score'], x['target'], x['source']))
    return {'total': len(records), 'records': records,
        'inventory_size': len(by_id), 'term_features_total': sum(map(len, profiles.values())),
        'term_features_by_source': {key: len(value) for key, value in sorted(profiles.items())},
        'term_feature_cap': None, 'context_source_regions': context_regions, 'explicit_label_pages': explicit_pages,
        'use_evidence_limit': _EVIDENCE_LIMIT,
        'statement_feature_policy': 'numbered names plus literal source-named/declaration constructions only',
        'full_source_reference_count': len(ledger[0]) if ledger is not None else None,
        'full_source_reference_digest': ledger[1] if ledger is not None else None,
        'retrieval_version': RETRIEVAL_VERSION, 'retrieval_parameters': dict(RETRIEVAL_PARAMETERS),
        'channel_candidate_counts': dict(sorted(Counter(channel for record in records
                                        for channel in record['retrieval_channels']).items())),
        'search_scope': 'all accepted nodes, all finite surviving source terms, all explicit pages, declared context ancestry, all distances including forward',
        'notice': 'Candidates are not dependencies. BM25 scores and local reading envelopes are retrieval clues, not actual-use evidence. Explicit occurrences are retained without top-k; lexical BM25 and local supplements are bounded. Context use_span is setup source, not target-node text or automatic edge evidence. Verify both source meaning and target use, and expand lexical misses.'}


def discover(project, draft, node_id=None, offset=0, limit=50):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('invalid relation candidate pagination')
    nodes = sorted((node for node in draft.data['nodes'].values() if node['judgment'] == 'accepted'),
                   key=lambda node: node['id'])
    if node_id is not None and node_id not in {node['id'] for node in nodes}:
        raise ValueError('candidate target must be an accepted registered node')
    # A content-bound cache cannot mask mutation of the pinned source files.
    # Historical adapters lacking this check are deliberately not cached.
    cacheable = hasattr(project, 'assert_unchanged') and hasattr(project, 'fingerprint')
    if cacheable:
        project.assert_unchanged()
    binding = _content_binding(project, draft, nodes)
    cache = getattr(project, '_relation_candidate_cache', None)
    if cache is None:
        cache = OrderedDict()
        project._relation_candidate_cache = cache
        project._relation_candidate_cache_stats = {'builds': 0, 'hits': 0}
    if cacheable and binding in cache:
        result = cache.pop(binding)
        cache[binding] = result
        project._relation_candidate_cache_stats['hits'] += 1
    else:
        result = _build_candidates(project, draft, nodes)
        project._relation_candidate_cache_stats['builds'] += 1
        if cacheable:
            cache[binding] = result
            while len(cache) > RETRIEVAL_PARAMETERS['cache_entries']:
                cache.popitem(last=False)
    records = result['records']
    if node_id is not None:
        records = [record for record in records if record['target'] == node_id]
    selected = records[offset:offset+limit]
    # Tool/WorkflowGate adds candidate IDs and binding fields after this call.
    # Never allow those mutations to alter the cached canonical inventory.
    return copy.deepcopy({**result, 'total': len(records), 'records': selected,
        'next_offset': offset + len(selected) if offset + len(selected) < len(records) else None,
        'candidate_content_sha256': binding})
