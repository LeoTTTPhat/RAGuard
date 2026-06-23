"""
RAGentGuard runtime configuration.
All thresholds, model choices, and behavioral flags live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PolicyConfig:
    """
    Thresholds for RAGentGuard's two-dimensional risk model.

    Risk dimension 1 — injection_risk (content-level, high-precision):
      Triggered when structural injection patterns are detected in the context.
      This is the PRIMARY signal for blocking.  In practice most benign external
      documents (Wikipedia, PDFs, web pages) contain no injection patterns, so
      injection_risk has a low false-positive rate even on fully untrusted corpora.

    Risk dimension 2 — trust_risk (source-level, contextual):
      Triggered when the fraction of context from low-trust sources exceeds a
      threshold.  This is a WARNING signal.  It should NOT block by default
      because a real RAG system routinely retrieves from unverified external
      sources; flagging every such query would yield FPR ≈ 100%.
      Set trust_only_blocking=True only in high-security deployments where every
      external source must be pre-approved.
    """
    # ---- injection_risk ------------------------------------------------ #
    # Minimum structural injection pattern matches to flag/block context
    injection_pattern_min_matches: int = 1

    # ---- trust_risk ---------------------------------------------------- #
    # Fraction of context from sources with trust_level < MEDIUM that triggers
    # a trust_risk flag (warning only unless trust_only_blocking=True)
    context_untrust_threshold: float = 0.7
    # Allow trust_risk alone (no injection patterns) to trigger a block.
    # Default False: trust_risk is warning-only.
    trust_only_blocking: bool = False

    # ---- tool-call ---------------------------------------------------- #
    # Taint score above which a tool call is blocked (requires injection pattern)
    tool_call_block_threshold: float = 0.5
    # Taint score above which a tool call is sent to human review
    tool_call_review_threshold: float = 0.3

    # ---- global -------------------------------------------------------- #
    # Master switch: if False, all violations are logged but never blocked
    blocking_mode: bool = True
    # Whether to include raw context snippets in audit reports (may be sensitive)
    include_context_snippets: bool = True


@dataclass
class AttributionConfig:
    """Settings for generation-time attribution."""
    method: str = "attention_rollout"    # "attention_rollout" | "gradient" | "random_baseline"
    top_k_sources: int = 5               # how many contributing sources to report
    min_attribution_score: float = 0.05  # ignore sources below this threshold


@dataclass
class VectorDBConfig:
    """Connection settings for the vector database."""
    backend: str = "chroma"              # "chroma" | "weaviate" | "pinecone" | "in_memory"
    collection_name: str = "ragentguard_docs"
    persist_directory: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class LLMConfig:
    """LLM backbone settings. Configure for any OpenAI-compatible or local endpoint."""
    provider: str = "openai_compatible"   # "openai_compatible" | "ollama" | "custom"
    model: str = "your-model-here"
    temperature: float = 0.0
    max_tokens: int = 2048
    api_key: Optional[str] = None        # if None, reads from RAGENTGUARD_API_KEY env var


@dataclass
class RAGentGuardConfig:
    """Top-level configuration aggregating all sub-configs."""
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Audit log output path
    audit_log_path: str = "./ragentguard_audit.jsonl"
    # HTML report output path
    audit_report_path: str = "./ragentguard_report.html"
    # Verbose logging
    verbose: bool = False
    # Trusted source IDs that bypass taint checks (e.g. curated knowledge bases)
    trusted_source_ids: List[str] = field(default_factory=list)
    # Optional local HMAC key for provenance manifests. This is not a full PKI
    # attestation system, but it prevents unsigned metadata from being promoted
    # to HIGH/SYSTEM trust when require_signed_high_trust=True.
    provenance_signing_key: Optional[str] = None
    require_signed_high_trust: bool = False

    @classmethod
    def default(cls) -> "RAGentGuardConfig":
        return cls()

    @classmethod
    def strict(cls) -> "RAGentGuardConfig":
        """High-security configuration: lower thresholds, always blocking."""
        cfg = cls()
        cfg.policy.context_untrust_threshold = 0.1
        cfg.policy.tool_call_block_threshold = 0.3
        cfg.policy.tool_call_review_threshold = 0.1
        cfg.policy.blocking_mode = True
        return cfg

    @classmethod
    def research_mode(cls) -> "RAGentGuardConfig":
        """Non-blocking config for evaluation experiments (logs everything, blocks nothing)."""
        cfg = cls()
        cfg.policy.blocking_mode = False
        cfg.attribution.method = "attention_rollout"
        return cfg
