"""
Stage 4: Generation-Time Attribution.

Maps each generated token/span back to the context segments it was most
influenced by.  Three methods are available:

  - "attention_rollout"  : uses attention weights from the last decoder layer
                           (requires transformers model with output_attentions=True)
  - "gradient"           : uses gradient × input (requires autograd access)
  - "overlap"            : lightweight string-overlap baseline (no model access)

For API-only (closed-weight) language models, only "overlap" is available.
The overlap method is used as the research baseline when comparing with TracLLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import AttributionConfig
from ..core.provenance import PipelineStage, ProvenanceTag, TaintVector, TrustLevel


@dataclass
class SpanAttribution:
    """Attribution score for one context span."""
    source_id: str
    chunk_id: str
    source_path: str
    trust_level: TrustLevel
    attribution_score: float          # 0.0 – 1.0
    matched_ngrams: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "trust_level": float(self.trust_level),
            "attribution_score": self.attribution_score,
            "matched_ngrams": self.matched_ngrams[:5],  # limit for report size
        }


@dataclass
class AttributionResult:
    """Full attribution analysis for one generation output."""
    generated_text: str
    attributions: List[SpanAttribution] = field(default_factory=list)
    method_used: str = "overlap"
    updated_taint: Optional[TaintVector] = None  # taint refined by attribution

    @property
    def top_source(self) -> Optional[SpanAttribution]:
        if not self.attributions:
            return None
        return max(self.attributions, key=lambda a: a.attribution_score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_text_preview": self.generated_text[:256],
            "method": self.method_used,
            "attributions": [a.to_dict() for a in self.attributions],
            "top_source_id": self.top_source.source_id if self.top_source else None,
        }


def _ngrams(text: str, n: int) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _overlap_score(generated: str, context_chunk: str, n: int = 3) -> Tuple[float, List[str]]:
    """
    Character n-gram overlap between generated text and a context chunk.
    Returns (score, matched_ngrams).
    """
    gen_ngrams = set(_ngrams(generated, n))
    ctx_ngrams = set(_ngrams(context_chunk, n))
    if not gen_ngrams:
        return 0.0, []
    matched = gen_ngrams & ctx_ngrams
    score = len(matched) / len(gen_ngrams)
    return score, list(matched)


class GenerationAttributor:
    """
    Maps generated output to source documents via one of three methods.
    Updates the TaintVector with attribution-weighted taint scores.
    """

    def __init__(self, config: AttributionConfig):
        self.config = config

    def attribute(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        model: Any = None,          # transformers model if method != "overlap"
        tokenizer: Any = None,
    ) -> AttributionResult:
        """
        Attribute generated_text to source chunks.

        Args:
            generated_text: LLM output to attribute
            chunk_taints: list of (chunk_text, TaintVector) from retrieval stage
            model: optional transformers model (for attention_rollout / gradient)
            tokenizer: optional tokenizer

        Returns AttributionResult with per-source attribution scores.
        """
        if self.config.method == "attention_rollout" and model is not None:
            return self._attention_rollout(generated_text, chunk_taints, model, tokenizer)
        if self.config.method == "gradient" and model is not None:
            return self._gradient_attribution(generated_text, chunk_taints, model, tokenizer)
        return self._overlap_attribution(generated_text, chunk_taints)

    # ------------------------------------------------------------------ #
    # Overlap baseline (always available, no model required)              #
    # ------------------------------------------------------------------ #

    def _overlap_attribution(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
    ) -> AttributionResult:
        attributions: List[SpanAttribution] = []

        for chunk_text, tv in chunk_taints:
            score, matched = _overlap_score(generated_text, chunk_text, n=3)
            if score < self.config.min_attribution_score:
                continue
            for tag in tv.contributing_tags:
                attributions.append(SpanAttribution(
                    source_id=tag.source_id,
                    chunk_id=tag.chunk_id,
                    source_path=tag.source_path,
                    trust_level=tag.trust_level,
                    attribution_score=score,
                    matched_ngrams=matched,
                ))

        # Keep top-k by score
        attributions.sort(key=lambda a: a.attribution_score, reverse=True)
        attributions = attributions[: self.config.top_k_sources]

        updated_taint = self._build_attributed_taint(attributions)
        return AttributionResult(
            generated_text=generated_text,
            attributions=attributions,
            method_used="overlap",
            updated_taint=updated_taint,
        )

    # ------------------------------------------------------------------ #
    # Attention rollout (requires transformers model)                      #
    # ------------------------------------------------------------------ #

    def _attention_rollout(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        model: Any,
        tokenizer: Any,
    ) -> AttributionResult:
        """
        Attention rollout over the decoder's self-attention layers.
        Falls back to overlap if torch is not available.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            return self._overlap_attribution(generated_text, chunk_taints)

        try:
            import torch

            context_text = "\n\n".join(t for t, _ in chunk_taints)
            prompt = f"{context_text}\n\nGenerated: {generated_text}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # Average attention across heads and layers
            attn_matrices = outputs.attentions  # tuple of (batch, heads, seq, seq)
            # Simple rollout: product of mean-head attention across layers
            rollout = torch.eye(attn_matrices[0].shape[-1])
            for layer_attn in attn_matrices:
                mean_attn = layer_attn[0].mean(dim=0)  # (seq, seq)
                rollout = rollout @ mean_attn

            # Map attention mass to each context chunk via token positions
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attributions = self._map_attn_to_chunks(
                rollout, tokens, chunk_taints, generated_text, tokenizer
            )
            attributions.sort(key=lambda a: a.attribution_score, reverse=True)
            attributions = attributions[: self.config.top_k_sources]

            return AttributionResult(
                generated_text=generated_text,
                attributions=attributions,
                method_used="attention_rollout",
                updated_taint=self._build_attributed_taint(attributions),
            )
        except Exception:
            # Any error → fall back to overlap
            return self._overlap_attribution(generated_text, chunk_taints)

    def _map_attn_to_chunks(
        self,
        rollout: Any,
        tokens: List[str],
        chunk_taints: List[Tuple[str, TaintVector]],
        generated_text: str,
        tokenizer: Any,
    ) -> List[SpanAttribution]:
        """Map rollout attention to source chunks by approximate token alignment."""
        import torch

        # Find generated text tokens (last part of sequence)
        gen_tokens = tokenizer(generated_text, add_special_tokens=False)["input_ids"]
        total_len = rollout.shape[-1]
        gen_start = max(0, total_len - len(gen_tokens))

        attributions: List[SpanAttribution] = []
        context_start = 0

        for chunk_text, tv in chunk_taints:
            chunk_tokens = tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
            chunk_end = min(context_start + len(chunk_tokens), gen_start)

            if chunk_end <= context_start:
                context_start += len(chunk_tokens)
                continue

            # Attention mass from generated tokens → this chunk's tokens
            gen_attn = rollout[gen_start:, context_start:chunk_end]
            score = float(gen_attn.mean().item()) if gen_attn.numel() > 0 else 0.0

            for tag in tv.contributing_tags:
                attributions.append(SpanAttribution(
                    source_id=tag.source_id,
                    chunk_id=tag.chunk_id,
                    source_path=tag.source_path,
                    trust_level=tag.trust_level,
                    attribution_score=score,
                ))

            context_start += len(chunk_tokens)

        return attributions

    # ------------------------------------------------------------------ #
    # Gradient attribution (requires autograd)                             #
    # ------------------------------------------------------------------ #

    def _gradient_attribution(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        model: Any,
        tokenizer: Any,
    ) -> AttributionResult:
        """
        Gradient × input attribution.
        Falls back to overlap if torch not available.
        """
        try:
            import torch

            context_text = "\n\n".join(t for t, _ in chunk_taints)
            full_text = f"{context_text}\n\nAnswer: {generated_text}"

            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            input_embeds = model.get_input_embeddings()(inputs["input_ids"])
            input_embeds = input_embeds.requires_grad_(True)

            outputs = model(inputs_embeds=input_embeds, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()

            grads = input_embeds.grad  # (1, seq, hidden)
            scores = (grads * input_embeds).sum(dim=-1).abs()[0]  # (seq,)

            attributions = self._map_gradient_to_chunks(
                scores, chunk_taints, context_text, generated_text, tokenizer
            )
            attributions.sort(key=lambda a: a.attribution_score, reverse=True)
            attributions = attributions[: self.config.top_k_sources]

            return AttributionResult(
                generated_text=generated_text,
                attributions=attributions,
                method_used="gradient",
                updated_taint=self._build_attributed_taint(attributions),
            )
        except Exception:
            return self._overlap_attribution(generated_text, chunk_taints)

    def _map_gradient_to_chunks(
        self,
        token_scores: Any,
        chunk_taints: List[Tuple[str, TaintVector]],
        context_text: str,
        generated_text: str,
        tokenizer: Any,
    ) -> List[SpanAttribution]:
        """Average gradient score over each chunk's token span."""
        attributions: List[SpanAttribution] = []
        offset = 0

        for chunk_text, tv in chunk_taints:
            chunk_token_ids = tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
            end = min(offset + len(chunk_token_ids), len(token_scores))
            chunk_scores = token_scores[offset:end]
            score = float(chunk_scores.mean().item()) if len(chunk_scores) > 0 else 0.0
            # Normalise to [0, 1]
            max_score = float(token_scores.max().item()) or 1.0
            score = score / max_score

            for tag in tv.contributing_tags:
                attributions.append(SpanAttribution(
                    source_id=tag.source_id,
                    chunk_id=tag.chunk_id,
                    source_path=tag.source_path,
                    trust_level=tag.trust_level,
                    attribution_score=score,
                ))
            offset += len(chunk_token_ids)

        return attributions

    # ------------------------------------------------------------------ #
    # Helper                                                               #
    # ------------------------------------------------------------------ #

    def _build_attributed_taint(
        self, attributions: List[SpanAttribution]
    ) -> TaintVector:
        """Build a TaintVector weighted by attribution scores."""
        if not attributions:
            return TaintVector()

        tags: List[ProvenanceTag] = []
        total_score = sum(a.attribution_score for a in attributions) or 1.0
        weighted_untrust = 0.0

        for attr in attributions:
            tag = ProvenanceTag(
                source_id=attr.source_id,
                chunk_id=attr.chunk_id,
                trust_level=attr.trust_level,
                source_path=attr.source_path,
            )
            tags.append(tag)
            weighted_untrust += attr.attribution_score * (1.0 - float(attr.trust_level))

        taint_score = weighted_untrust / total_score

        return TaintVector(
            contributing_tags=tags,
            taint_score=min(1.0, taint_score),
            propagation_path=[
                PipelineStage.INGESTION,
                PipelineStage.RETRIEVAL,
                PipelineStage.CONTEXT_ASSEMBLY,
                PipelineStage.GENERATION,
            ],
        )
