"""
Stage 5: Tool-Call Taint Checker.

Intercepts tool calls before execution, inspects their taint provenance,
and blocks or flags calls that carry high taint from adversarial documents.

This is the last line of defence before an adversarial document's influence
manifests as a real-world action (code execution, web fetch, email send, etc.).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

from ..core.config import RAGentGuardConfig
from ..core.policy import PolicyEngine
from ..core.provenance import PipelineStage, PolicyViolation, TaintVector

logger = logging.getLogger(__name__)


class ToolCallMonitor:
    """
    Wraps PolicyEngine for Stage-5 evaluation.

    Usage (direct):
        monitor = ToolCallMonitor(config)
        violation = monitor.check("bash", {"command": "rm -rf /"}, taint, query)
        if violation and violation.blocked:
            raise PolicyBlockedError(violation)

    Usage (decorator):
        @monitor.guard(current_taint)
        def call_bash(command: str) -> str:
            ...
    """

    def __init__(self, config: RAGentGuardConfig):
        self.config = config
        self.engine = PolicyEngine(config.policy)
        self._violations: List[PolicyViolation] = []

    def check(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        taint: TaintVector,
        query: str = "",
    ) -> Optional[PolicyViolation]:
        """
        Evaluate a tool call before execution.

        Returns PolicyViolation if taint exceeds threshold, else None.
        """
        violation = self.engine.evaluate_tool_call(tool_name, tool_args, taint, query)
        if violation:
            self._violations.append(violation)
            level = logging.ERROR if violation.blocked else logging.WARNING
            logger.log(
                level,
                "[RAGentGuard Stage 5] Tool call %s %s: %s",
                tool_name,
                "BLOCKED" if violation.blocked else "FLAGGED",
                violation.violation_type,
            )
        return violation

    def guard(
        self,
        taint: TaintVector,
        query: str = "",
    ) -> Callable:
        """
        Decorator factory.  Wraps a callable tool with pre-call taint checks.

        Example:
            @monitor.guard(current_taint, query="user question")
            def send_email(to: str, body: str) -> bool: ...
        """
        def decorator(fn: Callable) -> Callable:
            tool_name = fn.__name__

            @functools.wraps(fn)
            def wrapper(*args, **kwargs) -> Any:
                # Combine positional + keyword args for inspection
                all_args: Dict[str, Any] = {**kwargs}
                import inspect
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                for i, val in enumerate(args):
                    if i < len(params):
                        all_args[params[i]] = val

                violation = self.check(tool_name, all_args, taint, query)
                if violation and violation.blocked:
                    raise ToolCallBlockedError(violation)
                return fn(*args, **kwargs)

            return wrapper
        return decorator

    @property
    def all_violations(self) -> List[PolicyViolation]:
        return list(self._violations)

    def reset(self) -> None:
        self._violations.clear()


class ToolCallBlockedError(Exception):
    """Raised when a tool-call taint check blocks execution."""

    def __init__(self, violation: PolicyViolation):
        self.violation = violation
        tool = (violation.tool_call or {}).get("tool", "unknown")
        super().__init__(
            f"RAGentGuard blocked tool call '{tool}': {violation.violation_type}"
        )
