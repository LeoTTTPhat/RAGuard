"""
Stage 2: Retrieval-Time Taint Propagation.

Intercepts the top-K retrieval results from the vector DB and attaches a
TaintVector to each retrieved chunk. When context is assembled (Stage 3),
the union of taint vectors is carried forward through the pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..core.provenance import (
    PipelineStage,
    ProvenanceTag,
    TaintVector,
    TrustLevel,
    verify_manifest_signature,
)
from ..core.config import RAGentGuardConfig


def _tag_from_metadata(metadata: Dict[str, Any]) -> Optional[ProvenanceTag]:
    """Reconstruct a ProvenanceTag from vector DB metadata dict, if present."""
    if "provenance" in metadata:
        try:
            return ProvenanceTag.from_dict(metadata["provenance"])
        except Exception:
            pass
    # Fallback: build a partial tag from loose fields
    if "source_id" in metadata:
        return ProvenanceTag(
            source_id=metadata.get("source_id", "unknown"),
            chunk_id=metadata.get("chunk_id", "unknown"),
            trust_level=TrustLevel(float(metadata.get("trust_level", 0.0))),
            source_path=metadata.get("source_path", ""),
        )
    return None


def _tag_has_valid_manifest_signature(tag: ProvenanceTag, config: RAGentGuardConfig) -> bool:
    manifest = tag.metadata.get("provenance_manifest", {})
    signature = tag.metadata.get("provenance_signature", "")
    if not verify_manifest_signature(manifest, signature, config.provenance_signing_key or ""):
        return False
    return (
        manifest.get("source_id") == tag.source_id
        and manifest.get("chunk_id") == tag.chunk_id
        and manifest.get("source_hash") == tag.source_hash
        and manifest.get("source_path") == tag.source_path
        and float(manifest.get("trust_level", -1.0)) == float(tag.trust_level)
    )


def _taint_score_from_tag(tag: ProvenanceTag, config: RAGentGuardConfig) -> float:
    """
    Compute a scalar taint score [0, 1] for a single document tag.
    Trusted sources → low score; untrusted → high score.
    Sources explicitly whitelisted in config → 0 unless signed high-trust
    provenance is required and the tag has no valid local manifest signature.
    """
    effective_trust = tag.trust_level
    unsigned_elevated_or_whitelisted = False
    if config.require_signed_high_trust:
        signed = _tag_has_valid_manifest_signature(tag, config)
        if not signed and (
            float(tag.trust_level) >= float(TrustLevel.HIGH)
            or tag.source_id in config.trusted_source_ids
        ):
            effective_trust = TrustLevel.LOW
            unsigned_elevated_or_whitelisted = True

    if tag.source_id in config.trusted_source_ids:
        if unsigned_elevated_or_whitelisted:
            return 1.0 - float(effective_trust)
        return 0.0
    # Invert trust level: TrustLevel(1.0) → score 0.0; TrustLevel(0.0) → score 1.0
    return 1.0 - float(effective_trust)


class TaintPropagator:
    """
    Wraps a retrieval function and propagates taint from retrieved chunks.

    Can be used standalone or as a wrapper around a LangChain retriever.
    """

    def __init__(self, config: RAGentGuardConfig):
        self.config = config

    def propagate(
        self,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Tuple[str, TaintVector]]:
        """
        Given a list of raw retrieval results (each with 'text' and 'metadata'),
        return a list of (text, TaintVector) pairs.

        The TaintVector for each chunk reflects the trust level of its source
        and records that it has been through the RETRIEVAL stage.
        """
        result: List[Tuple[str, TaintVector]] = []

        for chunk in retrieved_chunks:
            text = chunk.get("text", chunk.get("page_content", ""))
            metadata = chunk.get("metadata", {})

            tag = _tag_from_metadata(metadata)

            if tag is None:
                # No provenance information: treat as fully untrusted
                tag = ProvenanceTag(
                    source_id="unknown-no-provenance",
                    trust_level=TrustLevel.UNTRUSTED,
                )

            taint_score = _taint_score_from_tag(tag, self.config)

            tv = TaintVector(
                contributing_tags=[tag],
                taint_score=taint_score,
                propagation_path=[PipelineStage.INGESTION, PipelineStage.RETRIEVAL],
            )

            result.append((text, tv))

        return result

    def assemble_context(
        self,
        tainted_chunks: List[Tuple[str, TaintVector]],
        separator: str = "\n\n",
    ) -> Tuple[str, TaintVector]:
        """
        Stage 2→3 handoff: concatenate chunk texts and merge taint vectors
        into a single context-level TaintVector.

        Returns (assembled_context_text, merged_taint_vector).
        """
        texts = [text for text, _ in tainted_chunks]
        assembled = separator.join(texts)

        merged = TaintVector()
        for _, tv in tainted_chunks:
            merged = merged.merge(tv)

        merged = merged.advance_stage(PipelineStage.CONTEXT_ASSEMBLY)
        return assembled, merged


class LangChainRetrieverWrapper:
    """
    Thin adapter that wraps a LangChain BaseRetriever with taint propagation.

    Usage:
        from langchain.vectorstores import Chroma
        retriever = Chroma(...).as_retriever()
        taint_retriever = LangChainRetrieverWrapper(retriever, config)
        tainted_chunks = taint_retriever.invoke("user query")
    """

    def __init__(self, retriever: Any, config: RAGentGuardConfig):
        self.retriever = retriever
        self.propagator = TaintPropagator(config)

    def invoke(self, query: str) -> List[Tuple[str, TaintVector]]:
        """Retrieve and immediately propagate taint."""
        docs = self.retriever.invoke(query)
        raw_chunks = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        return self.propagator.propagate(raw_chunks)

    def assemble_context(
        self,
        query: str,
    ) -> Tuple[str, TaintVector, List[Tuple[str, TaintVector]]]:
        """
        Full retrieval → context assembly in one call.

        Returns:
            assembled_text: str
            merged_taint: TaintVector
            chunk_taints: list of (text, TaintVector) per retrieved chunk
        """
        chunk_taints = self.invoke(query)
        assembled_text, merged_taint = self.propagator.assemble_context(chunk_taints)
        return assembled_text, merged_taint, chunk_taints
