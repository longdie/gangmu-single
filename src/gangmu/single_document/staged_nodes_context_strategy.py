"""Explicit two-round Context strategy candidate, without changing frozen v3.

This reuses the source host and response schemas. Only the instructions and
producer binding differ; an injected judge still owns all semantic decisions.
No old model responses or scores are transplanted onto this strategy.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from . import staged_nodes as base

VERSION = 'staged-nodes-v4-context-candidate'
POLICY = base.PROMPT_ROOT / 'context_policy_v4.md'


class _ContextPipeline(base._Pipeline):
    def __init__(self, project, rule_result, judge, options):
        super().__init__(project, rule_result, judge, options)
        base_binding = copy.deepcopy(self.binding)
        policy_raw = POLICY.read_bytes()
        policy = policy_raw.decode('utf-8')
        if not policy.strip():
            raise ValueError('The Context strategy requires a nonempty policy')
        self.prompts = {task: prompt.rstrip() + '\n\n' + policy
                        for task, prompt in self.prompts.items()}
        self.binding = {
            **base_binding,
            'producer_version': VERSION,
            'producer_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'base_producer_version': base_binding['producer_version'],
            'base_producer_code_sha256': base_binding['producer_code_sha256'],
            'base_prompt_sha256': base_binding['prompt_sha256'],
            'context_policy_sha256': hashlib.sha256(policy_raw).hexdigest(),
            'prompt_sha256': {task: hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                              for task, prompt in self.prompts.items()},
        }

    def run(self):
        result = super().run()
        result['producer_version'] = VERSION
        return result


def extract_complete_nodes(project, rule_result, judge: base.Judge, options=None):
    """Use the candidate in both existing rounds; this is not acceptance.

    No model client or ledger is constructed here. Callers must retain the
    existing stage, sample, budget and independent evaluation gates. The
    resulting producer binding distinguishes both actual prompt texts and
    both code modules from v3, including the cache request identity.
    """
    return _ContextPipeline(project, rule_result, judge,
                            base.NodeOptions.parse(options)).run()
