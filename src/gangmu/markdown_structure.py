"""Source-preserving Markdown blocks. No rendering, OCR, or text cleanup.

Heading/paragraph segmentation follows the old project's useful approach,
but uses original character ranges and protects code, comments and display math.
The index is a discovery aid, never an assertion of mathematical truth.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

VERSION = 'gangmu.markdown-structure.v1'
SUFFIXES = {'.md', '.markdown'}
FORMAL = re.compile(r'(?i)^(?:#{1,6}\s*)?(?:\*\*|__|\*)?\s*(?:Definition|Theorem|Lemma|Proposition|Corollary|Claim|Assertion|Conjecture|Proof|Notation|Remark|Example|Assumption|定义|定理|引理|命题|推论|证明)\b')
NUMBERED = re.compile(r'(?i)\b(Theorem|Lemma|Definition|Proposition|Corollary|Claim|Assertion|Conjecture|Assumption|Notation|Remark|Example|Equation)(?:\*\*|__)?\s+(\d+(?:\.\d+)*[a-z]?)\b')


def blocks(text, file='main.md'):
    """Partition every original character, including CRLF and blank lines."""
    lines = text.splitlines(keepends=True)
    result, i, offset = [], 0, 0
    while i < len(lines):
        start, first = offset, lines[i]
        stripped = first.strip()
        fence = re.match(r'^ {0,3}(`{3,}|~{3,})', first)
        environment = re.match(r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|displaymath)\}', stripped)
        math = stripped.startswith('$$') or stripped.startswith(r'\[') or environment
        kind, count, closed = 'paragraph', 1, True
        if not stripped:
            kind = 'blank'
        elif fence:
            kind, closed = 'code', False
            marker = fence[1]
            while i + count < len(lines):
                line = lines[i + count]
                count += 1
                if re.fullmatch(r' {0,3}' + re.escape(marker[0]) + '{' + str(len(marker)) + r',}[ \t]*\r?\n?', line):
                    closed = True
                    break
        elif stripped.startswith('<!--'):
            kind, closed = 'comment', '-->' in first
            while not closed and i + count < len(lines):
                closed = '-->' in lines[i + count]
                count += 1
        elif math:
            kind = 'math'
            end = (r'\end{' + environment[1] + '}') if environment else '$$' if stripped.startswith('$$') else r'\]'
            closed = end in stripped[2:]
            while not closed and i + count < len(lines):
                closed = end in lines[i + count]
                count += 1
        elif re.match(r'^ {0,3}#{1,6}\s+', first):
            kind = 'heading'
        elif i + 1 < len(lines) and re.fullmatch(r' {0,3}(?:=+|-+)[ \t]*\r?\n?', lines[i + 1]):
            kind, count = 'heading', 2
        else:
            while i + count < len(lines):
                line = lines[i + count]
                if (not line.strip() or re.match(r'^ {0,3}(?:#{1,6}\s|`{3,}|~{3,}|\$\$|\\\[|\\begin\{(?:equation|align|gather|multline|displaymath)|<!--)', line)
                        or FORMAL.match(line.strip())):
                    break
                count += 1
        offset += sum(len(line) for line in lines[i:i + count])
        literal = text[start:offset]
        row = {'id': f'md:{start}:{offset}', 'kind': kind, 'closed': closed,
               'source': {'file': file, 'char_start': start, 'char_end': offset},
               'formal_hint': bool(FORMAL.match(literal.strip())) if kind not in {'code', 'comment', 'math'} else False}
        if kind == 'heading':
            atx = re.match(r'^ {0,3}(#{1,6})\s+(.*)', first)
            row.update(level=len(atx[1]) if atx else (1 if lines[i+1].lstrip().startswith('=') else 2),
                       title=atx[2].strip().rstrip('#').strip() if atx else first.strip())
        result.append(row)
        i += count
    return result


def active_text(text):
    """Mask Markdown non-prose without moving offsets; percent is ordinary text."""
    chars = list(text)
    for block in blocks(text):
        if block['kind'] in {'code', 'comment'}:
            a, b = block['source']['char_start'], block['source']['char_end']
            chars[a:b] = ['\n' if c == '\n' else ' ' for c in text[a:b]]
    masked = ''.join(chars)
    for match in re.finditer(r'(`+)[^\n]*?\1', masked):
        chars[match.start():match.end()] = ' ' * len(match[0])
    return ''.join(chars)


def labels(text):
    active = active_text(text)
    result = []
    result.extend(m[1].strip() for m in re.finditer(r'\\label\s*\{([^{}]+)\}', active))
    result.extend(m[1] for m in re.finditer(r'\{#([\w:.-]+)\}', active))
    result.extend(m[1] for m in re.finditer(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', active, re.I))
    first = re.sub(r'^[#* _>]+', '', active).splitlines()[0] if active else ''
    match = NUMBERED.match(first)
    if match:
        result.append(match[1].lower() + ':' + match[2])
    result.extend('equation:' + m[1] for m in re.finditer(r'\\tag\s*\{([^{}]+)\}', active))
    return sorted(set(result))


def references(text):
    active, result = active_text(text), []
    patterns = [(NUMBERED, lambda m: m[1].lower() + ':' + m[2]),
                (re.compile(r'\\(?:eqref|ref|autoref|[cC]ref)\s*\{([^{}]+)\}'), lambda m: m[1]),
                (re.compile(r'\[[^\]]*\]\(#([^\s)]+)\)'), lambda m: m[1]),
                (re.compile(r'\((\d+(?:\.\d+)*)\)'), lambda m: 'equation:' + m[1])]
    for pattern, key in patterns:
        for match in pattern.finditer(active):
            for label in key(match).split(','):
                result.append({'command': 'markdown_reference', 'char_start': match.start(),
                    'char_end': match.end(), 'label': label.strip(), 'syntax': 'literal'})
    return sorted(result, key=lambda r: (r['char_start'], r['label']))


def extract(root, main_file, *, paper_id, arxiv_id):
    from gangmu.representation.sources import SourceError
    raw = (Path(root) / main_file).read_bytes()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as error:
        raise SourceError('Markdown input must be UTF-8; preserve and explicitly convert a new input version') from error
    if not text.strip():
        raise SourceError('Markdown main file is empty')
    records = blocks(text, main_file)
    sections = [{**row, 'activity': 'active', 'command': ('chapter', 'section', 'subsection', 'subsubsection', 'paragraph', 'subparagraph')[row['level']-1]}
                for row in records if row['kind'] == 'heading']
    structure = {'schema_version': VERSION, 'source_format': 'markdown', 'paper_id': paper_id,
                 'arxiv_id': arxiv_id, 'main_file': main_file, 'blocks': records, 'sections': sections}
    mapping = [{'source': {'file': main_file, 'char_start': 0, 'char_end': len(text)},
                'occurrence': 'markdown:main', 'reading_char_start': 0, 'reading_char_end': len(text)}]
    unresolved = [{'reason': 'unclosed_' + row['kind'], 'source': row['source']} for row in records if not row['closed']]
    quality = {'status': 'partial' if unresolved else 'complete', 'unresolved': unresolved,
               'source_roundtrip_failures': [], 'ocr_performed': False,
               'limits': ['Images are retained as references, not interpreted.', 'Only the selected main Markdown is the reading stream; other files are attachments.']}
    return structure, text, mapping, quality


def preprocess(root, main_file, output, *, paper_id, arxiv_id):
    structure, reading, mapping, quality = extract(root, main_file, paper_id=paper_id, arxiv_id=arxiv_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (('structure.json', structure), ('source_map.json', mapping), ('quality.json', quality)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'reading.md').write_bytes(reading.encode('utf-8'))
    return quality
