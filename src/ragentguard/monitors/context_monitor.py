"""
Stage 3: Context Policy Monitor.

Evaluates the assembled context window before it is submitted to the LLM.
Applies StruQ-inspired structural parsing to detect injection patterns
and computes the untrusted-source fraction of the context.

This is the first gate in RAGentGuard: if policy is violated and blocking_mode
is True, the LLM call is prevented and a PolicyViolation is returned instead.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..core.config import RAGentGuardConfig
from ..core.policy import PolicyEngine
from ..core.provenance import PipelineStage, PolicyViolation, TaintVector

logger = logging.getLogger(__name__)


class ContextPolicyMonitor:
    """
    Wraps PolicyEngine for Stage-3 evaluation.

    Typical call sequence:
        monitor = ContextPolicyMonitor(config)
        violation = monitor.check(assembled_context, chunk_taints, query)
        if violation and violation.blocked:
            raise PolicyBlockedError(violation)
        # otherwise proceed to LLM
    """

    def __init__(self, config: RAGentGuardConfig):
        self.config = config
        self.engine = PolicyEngine(config.policy)
        self._violations: List[PolicyViolation] = []

    def check(
        self,
        assembled_context: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        query: str = "",
    ) -> Optional[PolicyViolation]:
        """
        Evaluate context before LLM submission.

        Returns PolicyViolation if threshold exceeded, else None.
        If violation.blocked is True, caller should NOT submit to LLM.
        """
        violation = self.engine.evaluate_context(assembled_context, chunk_taints, query)
        if violation:
            self._violations.append(violation)
            if self.config.verbose:
                logger.warning(
                    "[RAGentGuard Stage 3] Violation detected: %s",
                    violation.violation_type,
                )
        return violation

    @property
    def all_violations(self) -> List[PolicyViolation]:
        return list(self._violations)

    def reset(self) -> None:
        self._violations.clear()


class PolicyBlockedError(Exception):
    """Raised when a context policy violation blocks LLM execution."""

    def __init__(self, violation: PolicyViolation):
        self.violation = violation
        super().__init__(
            f"RAGentGuard blocked LLM call: {violation.violation_type}"
        )
