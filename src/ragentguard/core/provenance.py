"""
Core data structures for RAGentGuard taint tracking.
Implements provenance tagging and taint propagation primitives.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TrustLevel(float, Enum):
    """Document trust tiers. Lower = less trusted."""
    UNTRUSTED = 0.0       # External/unverified source
    LOW = 0.25            # User-uploaded, unvalidated
    MEDIUM = 0.5          # Internal but unreviewed
    HIGH = 0.75           # Internal reviewed
    SYSTEM = 1.0          # Hardcoded system content


class AttackCategory(str, Enum):
    RETRIEVAL_INJECTION = "retrieval_injection"
    MEMORY_POISONING = "memory_poisoning"
    JUDGE_MANIPULATION = "judge_manipulation"
    CROSS_TOOL_TAINT = "cross_tool_taint"
    UNKNOWN = "unknown"


class PipelineStage(str, Enum):
    INGESTION = "ingestion"
    CHUNKING_EMBEDDING = "chunking_embedding"
    VECTOR_DB_STORAGE = "vector_db_storage"
    RETRIEVAL = "retrieval"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION = "generation"
    TOOL_DISPATCH = "tool_dispatch"


def canonical_manifest_bytes(manifest: Dict[str, Any]) -> bytes:
    """Serialize a provenance manifest deterministically for signing."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(manifest: Dict[str, Any], key: str) -> str:
    """Return an HMAC-SHA256 signature for a local provenance manifest."""
    return hmac.new(
        key.encode("utf-8"),
        canonical_manifest_bytes(manifest),
        hashlib.sha256,
    ).hexdigest()


def verify_manifest_signature(manifest: Dict[str, Any], signature: str, key: str) -> bool:
    """Verify a local HMAC-SHA256 provenance manifest signature."""
    if not manifest or not signature or not key:
        return False
    expected = sign_manifest(manifest, key)
    return hmac.compare_digest(expected, signature)


@dataclass
class ProvenanceTag:
    """
    Identity and integrity stamp assigned to each document chunk at ingestion time.
    Persisted alongside the embedding in vector DB metadata.
    """
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ingestion_time: datetime = field(default_factory=datetime.utcnow)
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    source_path: str = ""           # file path or URL
    source_hash: str = ""           # SHA-256 of raw document
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "ingestion_time": self.ingestion_time.isoformat(),
            "trust_level": float(self.trust_level),
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceTag":
        return cls(
            source_id=d["source_id"],
            chunk_id=d["chunk_id"],
            ingestion_time=datetime.fromisoformat(d["ingestion_time"]),
            trust_level=TrustLevel(d["trust_level"]),
            source_path=d.get("source_path", ""),
            source_hash=d.get("source_hash", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class TaintVector:
    """
    Accumulated taint state carried through the pipeline.
    Tracks which source documents influenced a given context or output.
    """
    contributing_tags: List[ProvenanceTag] = field(default_factory=list)
    taint_score: float = 0.0          # max propagated untrust; 0.0 = clean, 1.0 = fully tainted
    propagation_path: List[PipelineStage] = field(default_factory=list)
    injection_patterns_found: List[str] = field(default_factory=list)
    detected_attack_category: AttackCategory = AttackCategory.UNKNOWN

    @property
    def is_tainted(self) -> bool:
        return self.taint_score > 0.0

    @property
    def unique_source_ids(self) -> Set[str]:
        return {t.source_id for t in self.contributing_tags}

    def merge(self, other: "TaintVector") -> "TaintVector":
        """Union two taint vectors (used at context assembly)."""
        combined_tags = self.contributing_tags + [
            t for t in other.contributing_tags
            if t.chunk_id not in {x.chunk_id for x in self.contributing_tags}
        ]
        combined_score = max(self.taint_score, other.taint_score)
        combined_path = list(dict.fromkeys(self.propagation_path + other.propagation_path))
        combined_patterns = list(set(self.injection_patterns_found + other.injection_patterns_found))
        return TaintVector(
            contributing_tags=combined_tags,
            taint_score=combined_score,
            propagation_path=combined_path,
            injection_patterns_found=combined_patterns,
            detected_attack_category=(
                self.detected_attack_category
                if self.detected_attack_category != AttackCategory.UNKNOWN
                else other.detected_attack_category
            ),
        )

    def advance_stage(self, stage: PipelineStage) -> "TaintVector":
        """Return a copy with the given stage appended to the propagation path."""
        return TaintVector(
            contributing_tags=self.contributing_tags,
            taint_score=self.taint_score,
            propagation_path=self.propagation_path + [stage],
            injection_patterns_found=self.injection_patterns_found,
            detected_attack_category=self.detected_attack_category,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taint_score": self.taint_score,
            "contributing_sources": [t.to_dict() for t in self.contributing_tags],
            "propagation_path": [s.value for s in self.propagation_path],
            "injection_patterns": self.injection_patterns_found,
            "attack_category": self.detected_attack_category.value,
        }


@dataclass
class PolicyViolation:
    """
    Record of a detected policy boundary crossing.
    """
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stage: PipelineStage = PipelineStage.CONTEXT_ASSEMBLY
    violation_type: str = ""            # human-readable label
    taint_vector: TaintVector = field(default_factory=TaintVector)
    context_fraction: float = 0.0       # fraction of context from untrusted sources
    blocked: bool = False               # whether the pipeline was halted
    query: str = ""                     # triggering user query
    raw_context_snippet: str = ""       # first 512 chars of assembled context
    tool_call: Optional[Dict[str, Any]] = None   # if violation at tool dispatch

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage.value,
            "violation_type": self.violation_type,
            "taint_vector": self.taint_vector.to_dict(),
            "context_fraction": self.context_fraction,
            "blocked": self.blocked,
            "query": self.query,
            "raw_context_snippet": self.raw_context_snippet[:512],
            "tool_call": self.tool_call,
        }
