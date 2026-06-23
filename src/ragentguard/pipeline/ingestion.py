"""
Stage 1: Provenance Tagging at Ingestion.

Intercepts document ingestion into the RAG corpus and attaches a ProvenanceTag
to each chunk. The tag is stored in vector DB metadata alongside the embedding,
making taint tracking possible at retrieval time.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.provenance import PipelineStage, ProvenanceTag, TrustLevel, sign_manifest


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """Simple sliding-window text chunker (fallback if no framework splitter)."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _detect_trust_level(
    source_path: str,
    metadata: Dict[str, Any],
) -> TrustLevel:
    """
    Heuristic: assign trust level from source path or metadata.
    Override by setting metadata["trust_level"] explicitly.
    """
    if "trust_level" in metadata:
        return TrustLevel(float(metadata["trust_level"]))
    path_lower = source_path.lower()
    if path_lower.startswith("system://") or path_lower.startswith("internal://"):
        return TrustLevel.HIGH
    if path_lower.startswith("http://") or path_lower.startswith("https://"):
        return TrustLevel.LOW
    if path_lower.startswith("upload://") or path_lower.endswith(".pdf"):
        return TrustLevel.LOW
    return TrustLevel.UNTRUSTED


class DocumentIngestor:
    """
    Instruments the ingestion stage with provenance tagging.

    Usage (standalone, no LangChain):
        ingestor = DocumentIngestor()
        tagged_chunks = ingestor.ingest(raw_text, source_path="upload://doc.pdf")

    Usage (LangChain-compatible):
        Each tagged chunk carries .metadata["provenance"] which can be stored
        in ChromaDB / other vector DBs using their metadata dict support.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        trusted_source_ids: Optional[List[str]] = None,
        provenance_signing_key: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.trusted_source_ids: List[str] = trusted_source_ids or []
        self.provenance_signing_key = provenance_signing_key
        self._ingested_count = 0

    def ingest(
        self,
        text: str,
        source_path: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk and tag a single document.

        Returns a list of dicts:
            {
                "text": str,
                "provenance": ProvenanceTag,
                "metadata": dict   # ready for vector DB insertion
            }
        """
        meta = extra_metadata or {}
        trust = _detect_trust_level(source_path, meta)
        source_hash = _sha256(text)

        # Derive a stable source_id from hash so re-ingestion is idempotent
        source_id = f"src-{source_hash[:16]}"
        if source_id in self.trusted_source_ids:
            trust = TrustLevel.HIGH

        chunks = _chunk_text(text, self.chunk_size, self.overlap)
        tagged = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{source_id}-chunk-{i:04d}"
            chunk_hash = _sha256(chunk_text)
            chunk_metadata = {**meta, "chunk_index": i, "total_chunks": len(chunks)}
            manifest = {
                "source_id": source_id,
                "chunk_id": chunk_id,
                "source_hash": source_hash,
                "chunk_hash": chunk_hash,
                "source_path": source_path,
                "trust_level": float(trust),
                "chunk_index": i,
            }
            chunk_metadata["chunk_hash"] = chunk_hash
            chunk_metadata["provenance_manifest"] = manifest
            if self.provenance_signing_key:
                chunk_metadata["provenance_signature_alg"] = "HMAC-SHA256"
                chunk_metadata["provenance_signature"] = sign_manifest(
                    manifest,
                    self.provenance_signing_key,
                )

            tag = ProvenanceTag(
                source_id=source_id,
                ingestion_time=__import__("datetime").datetime.utcnow(),
                trust_level=trust,
                source_path=source_path,
                source_hash=source_hash,
                metadata=chunk_metadata,
            )
            # Override chunk_id with deterministic ID for reproducibility
            tag.chunk_id = chunk_id

            serialized_tag = tag.to_dict()

            tagged.append({
                "text": chunk_text,
                "provenance": tag,
                "metadata": {
                    "provenance": serialized_tag,
                    "source_path": source_path,
                    "trust_level": float(trust),
                    "chunk_index": i,
                    "source_id": source_id,
                },
            })

        self._ingested_count += len(tagged)
        return tagged

    def ingest_batch(
        self,
        documents: List[Tuple[str, str]],  # list of (text, source_path)
    ) -> List[Dict[str, Any]]:
        """Ingest multiple documents, returning all tagged chunks."""
        result = []
        for text, source_path in documents:
            result.extend(self.ingest(text, source_path))
        return result

    @property
    def total_ingested(self) -> int:
        return self._ingested_count


def langchain_ingest_hook(
    documents: List[Any],
    ingestor: Optional[DocumentIngestor] = None,
) -> List[Any]:
    """
    Compatibility shim for LangChain Document objects.
    Mutates document.metadata in-place to add provenance tags.

    Usage:
        from langchain.schema import Document
        docs = [Document(page_content="...", metadata={"source": "file.pdf"})]
        tagged_docs = langchain_ingest_hook(docs)
    """
    if ingestor is None:
        ingestor = DocumentIngestor()

    for doc in documents:
        source_path = doc.metadata.get("source", "")
        tagged_chunks = ingestor.ingest(doc.page_content, source_path, doc.metadata)
        if tagged_chunks:
            # Attach provenance of first chunk (whole doc if not split)
            doc.metadata["provenance"] = tagged_chunks[0]["metadata"]["provenance"]
            doc.metadata["trust_level"] = tagged_chunks[0]["metadata"]["trust_level"]
            doc.metadata["source_id"] = tagged_chunks[0]["metadata"]["source_id"]

    return documents
