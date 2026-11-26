"""Source-only Context interpretation inside the two existing node rounds.

The new policy changes the injected judge's instructions, not its response
schema or acceptance contract. Context review is part of the same production
judgment; it is not an independent semantic certificate or an extra round.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from gangmu.representation.model import digest
from . import staged_nodes as base
from . import staged_nodes_source_query_v1 as query
from .source import SourceError

VERSION = "staged-nodes-context-scope-v1"
POLICY = base.PROMPT_ROOT / "context_scope_v1.md"


class _ContextScopePipeline(query._RetrievalPipeline):
    def __init__(self, project, rule_result, judge, options, wire_options):
        if options.max_context_requests > 2:
            raise SourceError("Context scope allows at most two source followups per task")
        super().__init__(project, rule_result, judge, options, wire_options)
        if self.wire_options.max_wire_bytes > 160000:
            raise SourceError("Context scope wire bound must not exceed 160000 bytes")
        inherited = copy.deepcopy(self.binding)
        policy_raw = POLICY.read_bytes()
        policy = policy_raw.decode("utf-8")
        if not policy.strip():
            raise SourceError("Context scope requires a nonempty source policy")
        original_prompts = {
            base.TASK_ROUND1: (base.PROMPT_ROOT / "round1.md").read_text(encoding="utf-8"),
            base.TASK_ROUND2: (base.PROMPT_ROOT / "round2.md").read_text(encoding="utf-8"),
        }
        self.prompts = {task: prompt.rstrip() + "\n\n" + policy
                        for task, prompt in original_prompts.items()}
        policy_sha = hashlib.sha256(policy_raw).hexdigest()
        self.binding = {
            **inherited,
            "producer_version": VERSION,
            "producer_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parent_producer_binding": inherited,
            "parent_producer_binding_sha256": digest(inherited),
            "base_prompt_sha256": {
                task: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for task, prompt in original_prompts.items()
            },
            "context_policy_sha256": policy_sha,
            "context_scope_policy_sha256": policy_sha,
            "prompt_sha256": {
                task: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for task, prompt in self.prompts.items()
            },
            "context_scope_policy": {
                "review_inside_existing_response": True,
                "needs": ["own_task_meaning", "actual_standing_scope",
                          "direct_author_definitions_and_conventions"],
                "read_source_is_not_attached_context": True,
                "recursive_proof_dependency_closure": False,
                "additional_review_calls": 0,
                "independent_semantic_certificate": False,
                "response_schema": "unchanged_original_two_round_schema",
                "acceptance": "unchanged_original_source_and_context_checks",
                "max_source_followups": options.max_context_requests,
            },
            "regular_rounds": [1, 2],
            "new_strategy_development_required": True,
        }

    def run(self):
        result = super().run()
        result["producer_version"] = VERSION
        return result


def extract_complete_nodes(project, rule_result, judge, options=None, wire_options=None):
    """Use the new policy with the inherited bounded source-query loop."""
    return _ContextScopePipeline(project, rule_result, judge,
                                 base.NodeOptions.parse(options), wire_options).run()
