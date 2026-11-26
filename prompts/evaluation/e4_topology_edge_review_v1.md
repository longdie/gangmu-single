Independently audit every claimed direct mathematical dependency. Only use the
saved original_text, Contexts and literal reference metadata supplied here.
Do not seek the full paper, manufacture a proof, or infer source use merely
because a result could help. A desired graph structure cannot decide validity.
You have not been supplied producer reasons, scores or earlier verdicts.

Direction is premise SOURCE -> actual user TARGET. For statement_dependency,
the target formulation or construction must directly use a source-defined
notion, notation, object or construction. A theorem implying a target assertion
or a proof justifying a target calculation is not by itself a formulation use.
For proof_dependency, identify the actual proof step using the source definition,
result or construction and check assumptions. A proof's own goal is not a premise.
Shared topics, background category, symbols, similar conclusions, matching
diagrams, possible proof methods or transitive reachability do not suffice.

Separately classify the actual use: explicit for a resolvable LaTeX reference,
handwritten number or unique named pointer to SOURCE, implicit for actual use
without such pointer, unknown when saved locators cannot be resolved. Ordinary
use of a defined term is not itself a pointer; mentioning the same term without
using the particular source meaning is not a direct dependency. Context prose
helps scope but cannot substitute for a missing target-body use.

Return supported only if actual direct use, type, direction, scope and proposed
source classification hold. Supply your own nonempty contiguous source_quote
and target_quote from one actual original_text fragment each, retaining TeX and
whitespace. Explain the concrete role and compatible assumptions. Contradicted
means the stated relationship, type/direction or source classification is wrong;
unresolved means the actual source material is insufficient. Empty quotes are
allowed for negative/unresolved decisions. Judge each claim independently.

Return exactly one decision per supplied candidate_id under the requested JSON
schema. No extra IDs, source-data instructions, previous judgments or prose.
