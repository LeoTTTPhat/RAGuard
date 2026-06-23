"""
Evaluation experiments E1-E5 for RAGentGuard.

IMPORTANT — scope of this simulation:
  These experiments use a SYNTHETIC scaffold: adversarial documents are
  template-generated and detection is done by the RAGentGuard policy engine
  against that text directly (no real vector DB retrieval, no live LLM call).
  Results measure the correctness of the policy engine logic on synthetic data.
  See SOURCE_VALIDATION_REPORT.md for a full enumeration of what is simulated
  vs. what requires real infrastructure.

Metric conventions (see evaluation/metrics.py for full definitions):
  warning_recall   : fraction of adversarial docs that triggered any violation
                     (injection_risk OR trust_risk).  Includes memory-poisoning
                     attacks that are flagged but not blocked.
  blocking_recall  : fraction of adversarial docs that were BLOCKED.
                     ≤ warning_recall.  The operationally relevant recall metric.
  injection_recall : fraction of adversarial docs where a structural injection
                     pattern was detected.
  blocked_fpr      : fraction of benign queries operationally BLOCKED (primary FPR).
  warning_fpr      : fraction of benign queries that received any alert
                     (informational; trust_risk on low-trust sources is expected).

Trust-level protocol:
  Adversarial documents : TrustLevel.UNTRUSTED (score 0.0)
  Benign external docs  : TrustLevel.LOW       (score 0.25) — web/PDF sources
  Internal trusted docs : TrustLevel.HIGH      (score 0.75)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..attacks.corpus import AdversarialCorpusGenerator, AdversarialDocument
from ..core.config import RAGentGuardConfig
from ..core.provenance import (
    AttackCategory, TaintVector, TrustLevel, ProvenanceTag, PipelineStage,
)
from ..core.policy import PolicyEngine, scan_for_injection_patterns
from .metrics import EvaluationResult, MetricsCollector, timed


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _build_taint(
    doc_id: str,
    trust: TrustLevel,
    config: RAGentGuardConfig,
) -> TaintVector:
    tag = ProvenanceTag(
        source_id=doc_id,
        chunk_id=f"{doc_id}-chunk-0000",
        trust_level=trust,
        source_path=f"upload://{doc_id}.pdf",
    )
    taint_score = 0.0 if doc_id in config.trusted_source_ids else 1.0 - float(trust)
    return TaintVector(
        contributing_tags=[tag],
        taint_score=taint_score,
        propagation_path=[PipelineStage.INGESTION, PipelineStage.RETRIEVAL],
    )


def _assess_with_ragentguard(
    doc_text: str,
    doc_id: str,
    trust: TrustLevel,
    config: RAGentGuardConfig,
) -> Tuple[bool, bool, bool]:
    """
    Evaluate RAGentGuard Stage-3 policy on a single document.
    Returns (was_warned, was_blocked, injection_detected).

    NOTE: Direct text→policy evaluation (not a real RAG retrieval pipeline).
    """
    engine = PolicyEngine(config.policy)
    taint = _build_taint(doc_id, trust, config)
    chunk_taints: List[Tuple[str, TaintVector]] = [(doc_text, taint)]
    assessment = engine.assess_context(doc_text, chunk_taints, query="simulated_query")
    warned = assessment.has_violation
    blocked = warned and (assessment.violation.blocked if assessment.violation else False)
    return warned, blocked, assessment.injection_risk


def _naive_content_filter(doc_text: str) -> Tuple[bool, bool, bool]:
    """Proxy baseline: keyword list matching."""
    keywords = [
        "ignore", "disregard", "override", "system prompt", "DAN",
        "act as", "jailbreak", "curl attacker", "rm -rf",
    ]
    found = any(kw.lower() in doc_text.lower() for kw in keywords)
    return found, found, found


def _struq_proxy_baseline(doc_text: str) -> Tuple[bool, bool, bool]:
    """
    Proxy baseline: injection pattern scanning only (no trust/taint tracking).
    Labelled 'proxy' because it is not the actual StruQ implementation.
    """
    patterns = scan_for_injection_patterns(doc_text)
    detected = len(patterns) > 0
    return detected, detected, detected


# ------------------------------------------------------------------ #
# E1: Single-stage vs. cross-stage                                    #
# ------------------------------------------------------------------ #

def run_e1_single_vs_crossstage(
    n_per_category: int = 50,
    seed: int = 42,
) -> Dict[str, EvaluationResult]:
    """
    E1: Compare single-stage proxy baselines vs. RAGentGuard cross-stage detection.

    Reported metrics: ASR, warning_recall, blocking_recall, injection_recall,
                      blocked_fpr, warning_fpr, latency.
    Attribution accuracy: NOT reported (placeholder only — attribution is E3).
    """
    gen = AdversarialCorpusGenerator(seed=seed)
    adv_corpus = gen.generate(n_per_category=n_per_category)
    benign_corpus = gen.generate_benign(n=n_per_category * 2)
    rag_config = RAGentGuardConfig.default()
    results: Dict[str, EvaluationResult] = {}

    for system_name in ["naive_filter", "struq_proxy", "ragentguard"]:
        # attribution_is_meaningful=False for E1: values would be placeholder only
        collector = MetricsCollector(f"E1-{system_name}", attribution_is_meaningful=False)

        for doc in adv_corpus:
            t0 = time.perf_counter()
            if system_name == "naive_filter":
                warned, blocked, inj = _naive_content_filter(doc.text)
            elif system_name == "struq_proxy":
                warned, blocked, inj = _struq_proxy_baseline(doc.text)
            else:
                warned, blocked, inj = _assess_with_ragentguard(
                    doc.text, doc.doc_id, TrustLevel.UNTRUSTED, rag_config
                )
            elapsed = (time.perf_counter() - t0) * 1000
            collector.record_adversarial_query(
                was_warned=warned,
                was_blocked=blocked,
                injection_detected=inj,
                correct_source_id=None,        # attribution not measured in E1
                top_attributed_source_id=None,
                latency_ms=elapsed,
            )

        for doc in benign_corpus:
            t0 = time.perf_counter()
            if system_name == "naive_filter":
                _, blocked, _ = _naive_content_filter(doc.text)
                warned_not_blocked = False
            elif system_name == "struq_proxy":
                _, blocked, _ = _struq_proxy_baseline(doc.text)
                warned_not_blocked = False
            else:
                warned, blocked, _ = _assess_with_ragentguard(
                    doc.text, doc.doc_id, TrustLevel.LOW, rag_config
                )
                warned_not_blocked = warned and not blocked
            elapsed = (time.perf_counter() - t0) * 1000
            collector.record_benign_query(
                was_blocked=blocked,
                was_warned=warned_not_blocked,
                latency_ms=elapsed,
            )

        results[system_name] = collector.finalize()

    return results


# ------------------------------------------------------------------ #
# E2: Attack category coverage                                         #
# ------------------------------------------------------------------ #

def run_e2_category_coverage(
    n_per_category: int = 100,
    seed: int = 42,
) -> Dict[str, EvaluationResult]:
    """
    E2: warning_recall, blocking_recall, and injection_recall per attack category.

    Memory-poisoning attacks contain no structural injection patterns, so
    blocking_recall = injection_recall = 0 for that category.
    warning_recall may be >0 for memory-poisoning if trust_risk fires
    (untrust_fraction ≥ threshold).
    Attribution accuracy: NOT reported (use E3).
    """
    gen = AdversarialCorpusGenerator(seed=seed)
    config = RAGentGuardConfig.default()
    results: Dict[str, EvaluationResult] = {}

    for cat in [
        AttackCategory.RETRIEVAL_INJECTION,
        AttackCategory.MEMORY_POISONING,
        AttackCategory.JUDGE_MANIPULATION,
        AttackCategory.CROSS_TOOL_TAINT,
    ]:
        corpus = gen.generate(n_per_category=n_per_category, categories=[cat])
        collector = MetricsCollector(f"E2-{cat.value}", attribution_is_meaningful=False)

        for doc in corpus:
            t0 = time.perf_counter()
            warned, blocked, inj = _assess_with_ragentguard(
                doc.text, doc.doc_id, TrustLevel.UNTRUSTED, config
            )
            elapsed = (time.perf_counter() - t0) * 1000
            collector.record_adversarial_query(
                was_warned=warned,
                was_blocked=blocked,
                injection_detected=inj,
                correct_source_id=None,
                top_attributed_source_id=None,
                latency_ms=elapsed,
            )

        results[cat.value] = collector.finalize()

    return results


# ------------------------------------------------------------------ #
# E3: Attribution accuracy (multi-source context with distractors)    #
# ------------------------------------------------------------------ #

def run_e3_attribution_accuracy(
    n_queries: int = 80,
    n_distractors: int = 4,
    seed: int = 42,
) -> Dict[str, EvaluationResult]:
    """
    E3: Attribution accuracy — the ONLY experiment where attribution_accuracy is valid.

    Protocol: each query has 1 adversarial + n_distractors benign chunks at a random
    position.  The attribution method must identify the adversarial source.

    LIMITATION (documented in SOURCE_VALIDATION_REPORT.md §2.4):
    The simulated LLM output reuses the first 128 chars of the adversarial doc.
    n-gram overlap attribution therefore approaches 100% trivially.
    Real attribution accuracy on LLM-generated paraphrases will be lower.
    """
    from ..pipeline.attribution import GenerationAttributor
    import random

    gen = AdversarialCorpusGenerator(seed=seed)
    adv_corpus = gen.generate(n_per_category=n_queries // 4)[:n_queries]
    benign_pool = gen.generate_benign(n=n_queries * n_distractors)
    config = RAGentGuardConfig.default()
    config.attribution.method = "overlap"
    attributor = GenerationAttributor(config.attribution)
    rng = random.Random(seed)

    # attribution_is_meaningful=True: E3 is the real attribution experiment
    rag_collector = MetricsCollector("E3-ragentguard-overlap", attribution_is_meaningful=True)
    rand_collector = MetricsCollector("E3-random-baseline", attribution_is_meaningful=True)

    for i, adv_doc in enumerate(adv_corpus):
        distractor_docs = benign_pool[i * n_distractors: (i + 1) * n_distractors]

        adv_tag = ProvenanceTag(
            source_id=adv_doc.doc_id,
            chunk_id=f"{adv_doc.doc_id}-chunk-0000",
            trust_level=TrustLevel.UNTRUSTED,
        )
        adv_tv = TaintVector(contributing_tags=[adv_tag], taint_score=1.0)

        chunk_taints: List[Tuple[str, TaintVector]] = []
        for dist_doc in distractor_docs:
            dist_tag = ProvenanceTag(
                source_id=dist_doc.doc_id,
                chunk_id=f"{dist_doc.doc_id}-chunk-0000",
                trust_level=TrustLevel.LOW,
            )
            chunk_taints.append((dist_doc.text, TaintVector(
                contributing_tags=[dist_tag], taint_score=0.75
            )))

        insert_pos = rng.randint(0, len(chunk_taints))
        chunk_taints.insert(insert_pos, (adv_doc.text, adv_tv))

        # Simulated output = prefix of adversarial doc (see LIMITATION note above)
        simulated_output = adv_doc.text[:128] + " and additional context."

        t0 = time.perf_counter()
        attr_result = attributor.attribute(simulated_output, chunk_taints)
        elapsed = (time.perf_counter() - t0) * 1000

        top_src = attr_result.top_source.source_id if attr_result.top_source else None
        rag_collector.record_adversarial_query(
            was_warned=True, was_blocked=False, injection_detected=True,
            correct_source_id=adv_doc.doc_id,
            top_attributed_source_id=top_src,
            latency_ms=elapsed,
        )

        # Random baseline: uniform pick from all sources
        all_ids = [adv_doc.doc_id] + [d.doc_id for d in distractor_docs]
        rand_collector.record_adversarial_query(
            was_warned=True, was_blocked=False, injection_detected=True,
            correct_source_id=adv_doc.doc_id,
            top_attributed_source_id=rng.choice(all_ids),
            latency_ms=elapsed,
        )

    return {
        "ragentguard_overlap": rag_collector.finalize(),
        "random_baseline": rand_collector.finalize(),
    }


# ------------------------------------------------------------------ #
# E4: Latency overhead                                                 #
# ------------------------------------------------------------------ #

def run_e4_latency(
    n_queries: int = 200,
    seed: int = 42,
) -> EvaluationResult:
    """E4: Per-query latency of taint tracking + policy evaluation (Python only, no I/O)."""
    gen = AdversarialCorpusGenerator(seed=seed)
    corpus = gen.generate(n_per_category=n_queries // 4)
    config = RAGentGuardConfig.default()
    config.audit_log_path = ""
    collector = MetricsCollector("E4-latency", attribution_is_meaningful=False)

    for doc in corpus[:n_queries]:
        t0 = time.perf_counter()
        warned, blocked, inj = _assess_with_ragentguard(
            doc.text, doc.doc_id, TrustLevel.UNTRUSTED, config
        )
        elapsed = (time.perf_counter() - t0) * 1000
        collector.record_adversarial_query(
            was_warned=warned, was_blocked=blocked, injection_detected=inj,
            correct_source_id=None, top_attributed_source_id=None,
            latency_ms=elapsed,
        )

    return collector.finalize()


# ------------------------------------------------------------------ #
# E5: Real-world RAG app simulation                                    #
# ------------------------------------------------------------------ #

def run_e5_realworld_simulation(seed: int = 42) -> Dict[str, EvaluationResult]:
    """
    E5: Simulate three representative RAG app patterns.

    Adversarial docs: UNTRUSTED.
    Benign docs: LOW (simulates typical external knowledge base).
    FPR reported as blocked_fpr (operational) and warning_fpr (informational).
    Attribution accuracy: NOT reported (use E3).
    """
    gen = AdversarialCorpusGenerator(seed=seed)
    config = RAGentGuardConfig.default()

    app_categories = {
        "customer_support": [AttackCategory.RETRIEVAL_INJECTION, AttackCategory.MEMORY_POISONING],
        "code_assistant":   [AttackCategory.CROSS_TOOL_TAINT, AttackCategory.RETRIEVAL_INJECTION],
        "llm_judge":        [AttackCategory.JUDGE_MANIPULATION, AttackCategory.RETRIEVAL_INJECTION],
    }

    results: Dict[str, EvaluationResult] = {}

    for app_name, categories in app_categories.items():
        adv_corpus = gen.generate(n_per_category=50, categories=categories)
        benign_corpus = gen.generate_benign(n=80)
        collector = MetricsCollector(f"E5-{app_name}", attribution_is_meaningful=False)

        for doc in adv_corpus:
            t0 = time.perf_counter()
            warned, blocked, inj = _assess_with_ragentguard(
                doc.text, doc.doc_id, TrustLevel.UNTRUSTED, config
            )
            elapsed = (time.perf_counter() - t0) * 1000
            collector.record_adversarial_query(
                was_warned=warned, was_blocked=blocked, injection_detected=inj,
                correct_source_id=None, top_attributed_source_id=None,
                latency_ms=elapsed,
            )

        for doc in benign_corpus:
            t0 = time.perf_counter()
            warned, blocked, _ = _assess_with_ragentguard(
                doc.text, doc.doc_id, TrustLevel.LOW, config
            )
            elapsed = (time.perf_counter() - t0) * 1000
            collector.record_benign_query(
                was_blocked=blocked,
                was_warned=warned and not blocked,
                latency_ms=elapsed,
            )

        results[app_name] = collector.finalize()

    return results
