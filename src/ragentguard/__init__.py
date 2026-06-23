"""
RAGentGuard: End-to-End Security and Privacy Auditing for RAG-Agent Pipelines.

Quick start:
    from ragentguard import RAGentGuard

    guard = RAGentGuard()
    tagged = guard.ingest("document text", "upload://doc.pdf")
    chunk_taints = guard.retrieve(tagged)
    context, taint = guard.assemble(chunk_taints)
    guard.check_context(context, chunk_taints, query="user query")
    guard.save_report("./audit_report.html")
"""
from .ragentguard import RAGentGuard
from .core import (
    AttackCategory,
    PipelineStage,
    PolicyViolation,
    ProvenanceTag,
    TaintVector,
    TrustLevel,
    RAGentGuardConfig,
)
from .monitors import PolicyBlockedError, ToolCallBlockedError

__version__ = "0.1.0"
__all__ = [
    "RAGentGuard",
    "RAGentGuardConfig",
    "AttackCategory",
    "PipelineStage",
    "PolicyViolation",
    "ProvenanceTag",
    "TaintVector",
    "TrustLevel",
    "PolicyBlockedError",
    "ToolCallBlockedError",
]
