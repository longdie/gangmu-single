import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gangmu.latex_structure import StructureExtractor


class SourceTests(unittest.TestCase):
    def parse(self, text, files=None):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "main.tex").write_text(text)
        for p, value in (files or {}).items():
            f = root / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(value)
        return StructureExtractor(root, "main.tex").extract()

    def test_no_phantom_labels_inside_comments_definition_verbatim(self):
        s, text, mapping, q = self.parse(r'''
% \label{comment}
\newcommand{\foo}[1]{\label{definition-#1}}
\verb|\label{verb}|
\begin{verbatim}\label{block}\end{verbatim}
\begin{document}\label{real}\foo{x}\end{document}
''')
        self.assertEqual([x['keys'] for x in s['labels']], [['real']])
        self.assertEqual(s['definitions'][0]['argument_count'],1)
        self.assertIn('custom_macro_requires_expansion',q['unresolved_by_reason'])

    def test_includes_order_repeat_and_roundtrip_utf8(self):
        s, text, mapping, q = self.parse('α\\input{a}β\\input{a}γ', {'a.tex':'δ\\label{repeat}'})
        self.assertEqual(text,'α\\input{a}δ\\label{repeat}β\\input{a}δ\\label{repeat}γ')
        self.assertEqual(len(s['labels']),2)
        self.assertEqual(q['source_roundtrip_failures'],[])
        self.assertIn('repeat',q['duplicate_labels'])

    def test_conditions_and_dynamic_input(self):
        s, _, _, q = self.parse(r'\iffalse\label{no}\else\label{yes}\fi\newif\ifdraft\ifdraft\label{maybe}\fi\input{\path}')
        self.assertEqual([(x['keys'],x['activity']) for x in s['labels']], [(['yes'],'active'),(['maybe'],'conditional')])
        self.assertEqual(s['includes'][0]['status'],'dynamic_path')
        self.assertEqual(q['unclosed_conditionals'],[])

    def test_group_scope_restores_definition(self):
        s,_,_,q = self.parse(r'\newcommand{\foo}[1]{outer}{\renewcommand{\foo}[2]{inner}\foo{x}{y}}\foo{z}')
        self.assertEqual([len(x['arguments']) for x in s['macro_calls']],[2,1])

    def test_theorem_and_cross_file_environment(self):
        s,_,_,q = self.parse(r'\newtheorem{prop}{Proposition}[section]\begin{prop}[Title]\input{body}\end{prop}', {'body.tex':r'\label{result}'})
        self.assertEqual(s['environments'][0]['environment_kind'],'theorem')
        self.assertTrue(s['environments'][0]['closed'])
        self.assertEqual(s['labels'][0]['environment_id'],s['environments'][0]['id'])

    def test_nested_group_optional_braces_escaped_and_comments(self):
        s,_,_,q=self.parse('\\section[short {a]b}]{A {B \\} C}%z\n D}\\label{x}\n\\ref{x}')
        self.assertEqual(len(s['sections']),1)
        self.assertEqual(s['sections'][0]['short_title'],'short {a]b}')
        self.assertTrue(s['references'][0]['candidate_targets']['x'])

    def test_cycle_and_external_path_not_executed(self):
        s,_,_,q=self.parse(r'\input{a}\input{../outside}',{'a.tex':r'\input{main}'})
        self.assertIn('input_cycle',q['unresolved_by_reason'])
        self.assertEqual(s['includes'][-1]['status'],'not_found')

    def test_custom_primitive_definition_not_interpreted_as_structure(self):
        s,_,_,q=self.parse(r'\def\myref#1, #2{\ref{#1}\label{#2}}\label{live}\makeatletter\def\foo@bar#1{body}\makeatother')
        self.assertEqual(s['definitions'][0]['argument_count'],2)
        self.assertTrue(s['definitions'][0]['delimited_parameters'])
        self.assertEqual([x['name'] for x in s['definitions']],['myref','foo@bar'])
        self.assertEqual([x['keys'] for x in s['labels']],[['live']])

    def test_ignored_macro_argument_cannot_create_phantom_label(self):
        s,_,_,q=self.parse(r'\newcommand{\discard}[1]{}\discard{\label{phantom}}\label{live}')
        self.assertEqual([x['keys'] for x in s['labels']],[['live']])
        self.assertIn('macro_argument_structure_not_expanded',q['unresolved_by_reason'])

    def test_adjacent_inline_math_not_a_display_switch(self):
        s,_,_,q=self.parse(r'$x$$y$ and $$z$$')
        self.assertEqual([x['delimiter'] for x in s['math_spans']],['$','$','$$'])
        self.assertTrue(all(x['closed'] for x in s['math_spans']))

    def test_package_at_catcode_and_deferred_body_do_not_leak(self):
        s,_,_,q=self.parse(r'\usepackage{mine}\begin{document}\label{live}\end{document}', {'mine.sty':r'\newcommand\@foo{bar}\DeclareOption{x}{\label{phantom}}\if@compatibility\def\z{a}\else\def\z{b}\fi'})
        self.assertIn('@foo',[x['name'] for x in s['definitions']])
        self.assertEqual([x['keys'] for x in s['labels']],[['live']])
        self.assertEqual(q['unclosed_conditionals'],[])

    def test_byte_fallback_marked_not_claimed_as_author_encoding(self):
        tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.addCleanup(tmp.cleanup)
        p=Path(tmp.name);(p/'main.tex').write_bytes(b'\\section{A}\xa3\xac\\label{x}')
        s,_,_,q=StructureExtractor(p,'main.tex').extract()
        self.assertEqual(q['source_roundtrip_failures'],[])
        self.assertEqual(q['encoding_unresolved_files'],['main.tex'])
        self.assertTrue(s['files'][0]['undecoded_non_ascii_spans'])

    def test_invalid_control_name_and_bibliography_link(self):
        s,_,_,q=self.parse(r'\newcommand{\foo.bar}{X}\cite{A}\begin{thebibliography}{9}\bibitem[Alpha]{A} Entry\end{thebibliography}')
        self.assertIn('invalid_literal_control_sequence_name',q['unresolved_by_reason'])
        self.assertTrue(s['citations'][0]['candidate_bibliography_items']['A'])

    def test_savebox_wrapper_preserves_literal_nested_input(self):
        s,text,_,q=self.parse(r'''\newcommand{\boxinminipage}[1]{%
\sbox\mybox{#1}%
\settowidth\mywidth{\usebox{\mybox}}%
\begin{minipage}{\mywidth}\usebox{\mybox}\end{minipage}%
}\boxinminipage{\input{figure}}''',{'figure.tex':r'\begin{tikzpicture}x\end{tikzpicture}'})
        self.assertEqual(s['includes'][0]['status'],'resolved')
        self.assertIn('figure.tex',[x['path'] for x in s['files']])
        self.assertTrue(s['macro_calls'][0].get('argument_traversal_adapter'))

    def test_section_siblings_replace_context(self):
        s,_,_,q=self.parse(r'\section{A}\subsection{A1}\label{a}\section{B}\label{b}')
        self.assertEqual(s['labels'][0]['section_ids'],[s['sections'][0]['id'],s['sections'][1]['id']])
        self.assertEqual(s['labels'][1]['section_ids'],[s['sections'][2]['id']])

    def test_theorem_title_and_section_title_refs_are_live(self):
        s,_,_,q=self.parse(r'\newtheorem{Th}{Theorem}\begin{Th}[Theorem \ref{later}]x\end{Th}\section{See \cite{AIR}}\label{later}')
        self.assertEqual(s['references'][0]['argument_role'],'environment_title')
        self.assertTrue(s['references'][0]['candidate_targets']['later'])
        self.assertEqual(s['citations'][0]['argument_role'],'section_title')


if __name__=='__main__': unittest.main()
