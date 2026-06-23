# RAGuard

**End-to-End Security and Privacy Auditing for Retrieval-Augmented Generation Agent Pipelines**

---

## Overview

RAGuard is a holistic, taint-flow-based security audit framework for RAG-agent pipelines. Where existing defenses (StruQ, TracLLM, content filters) inspect only one stage in isolation, RAGuard tracks adversarial content across all **seven pipeline stages**:

```
Stage 1: Ingestion
Stage 2: Chunking / Embedding
Stage 3: Vector DB Storage
Stage 4: Retrieval
Stage 5: Context Assembly  ← Context Policy Monitor
Stage 6: Generation        ← Generation-Time Attribution
Stage 7: Tool Dispatch     ← Tool-Call Taint Checker
```

A malicious document injected into the corpus propagates a taint tag through every stage. RAGuard detects policy violations at each monitored boundary and produces a structured forensic audit report.

### Two-Dimensional Policy Model

RAGuard separates two distinct risk signals to avoid excessive false positives:

| Risk dimension | Signal | Default action |
|---|---|---|
| **injection_risk** | Structural injection patterns detected in context (StruQ-inspired) | **BLOCK** |
| **trust_risk** | High fraction of context from low-trust sources | **WARN** (not block by default) |

This separation is critical: a real RAG system routinely retrieves from unverified external sources (web, user PDFs). Blocking on `trust_risk` alone yields FPR ≈ 100% in the synthetic setup. Blocking on `injection_risk` alone is intended to reduce operational false positives, but must still be validated on real benign corpora.
The artifact therefore reports both `blocked_fpr` (operational failures) and `warning_fpr` (alert volume); warning recall should not be interpreted as blocking protection.

---

## Architecture

```
src/ragentguard/
├── core/
│   ├── provenance.py     # ProvenanceTag, TaintVector, PolicyViolation, AttackCategory
│   ├── policy.py         # PolicyEngine — two-dimensional risk assessment
│   └── config.py         # RAGentGuardConfig (thresholds, modes, trust flags)
├── pipeline/
│   ├── ingestion.py      # Stage 1: provenance tagging at document ingestion
│   ├── retrieval.py      # Stage 2: taint propagation over retrieved chunks
│   └── attribution.py    # Stage 4: generation-time attribution (overlap/attention/gradient)
├── monitors/
│   ├── context_monitor.py  # Stage 3: context policy gate (pre-LLM)
│   └── tool_monitor.py     # Stage 5: tool-call taint checker (pre-execution)
├── audit/
│   └── reporter.py       # Stage 6: JSON + HTML audit report generator
├── attacks/
│   └── corpus.py         # Synthetic adversarial corpus generator (4 categories)
└── evaluation/
    ├── metrics.py         # ASR; warning/blocking/injection recall; blocked_fpr/warning_fpr; attribution accuracy (E3 only); latency
    ├── experiments.py     # E1-E5 controlled experiment runners with labelled baselines
    └── end_to_end.py      # E6/E7 local RAG pipelines with SQLite/FAISS retrieval
```

---

## Threat Model

| Attack category | Description |
|---|---|
| **Retrieval injection** | Adversarial document retrieved for benign query hijacks downstream tool calls |
| **Memory poisoning** | Malicious content in long-term vector memory re-surfaces and accumulates |
| **Judge manipulation** | Adversarial retrieval biases LLM-as-a-judge evaluation output |
| **Cross-tool taint** | Tainted content from Stage A propagates through multi-hop tool chains to Stage C |

---

## Quick Start

```python
from ragentguard import RAGentGuard, RAGentGuardConfig
from ragentguard.monitors import PolicyBlockedError

config = RAGentGuardConfig.default()
guard = RAGentGuard(config)

# Stage 1 – Ingest (assigns provenance tag to each chunk)
chunks = guard.ingest("adversarial doc...", source_path="upload://user.pdf")

# Stage 2 – Retrieve + propagate taint
retrieved = guard.retrieve(chunks)

# Stage 2→3 – Assemble context
context, taint = guard.assemble(retrieved)

# Stage 3 – Context policy gate
try:
    guard.check_context(context, retrieved, query="user query")
except PolicyBlockedError as e:
    print("Blocked:", e.violation.violation_type)

# Stage 4 – Generation attribution
attribution = guard.attribute("generated output", retrieved)
print("Top source:", attribution.top_source.source_id)

# Stage 5 – Tool-call gate
guard.check_tool_call("bash", {"command": "ls"}, taint)

# Stage 6 – Audit report
guard.save_report("audit.html", "audit.json")
```

---

## Configuration

```python
# Default (production: blocking on injection patterns)
config = RAGentGuardConfig.default()

# Strict (also block on high untrusted-source fraction)
config = RAGentGuardConfig.strict()
config.policy.trust_only_blocking = True

# Research mode (non-blocking: all violations logged, nothing blocked)
config = RAGentGuardConfig.research_mode()

# Custom thresholds
config.policy.context_untrust_threshold = 0.5    # trust_risk threshold
config.policy.injection_pattern_min_matches = 1  # injection_risk threshold
config.policy.tool_call_block_threshold = 0.5    # tool-call taint threshold
```

---

## Evaluation Experiments

```bash
python examples/run_experiments.py
```

| Experiment | Description |
|---|---|
| **E1** | Single-stage baselines (naive filter, StruQ proxy) vs. RAGuard cross-stage detection |
| **E2** | Detection recall per attack category |
| **E3** | Attribution accuracy vs. random baseline (multi-source context with distractors) |
| **E4** | Per-query latency overhead of taint tracking |
| **E5** | Simulated real-world RAG app patterns (customer_support / code_assistant / llm_judge) |
| **E6** | End-to-end local RAG run with SQLite vector storage, ranked retrieval, context assembly, deterministic generation, and monitored tool dispatch |
| **E7** | Real-stack RAG run with FAISS, sentence-transformers, FLAN-T5-small generation, AG News/Wikitext passages, 68 FLAN-T5-mutated attacks, and same-context baselines |
| **E8** | Manual red-team attack set and sandboxed real-tool execution through the tool monitor |
| **E9** | Calibrated hybrid defense, tuned same-trace baseline parity, aggregation ablation, and scaled operator-cost study |

---

## Installation

```bash
pip install -e .                          # core only (no external deps)
pip install -e ".[langchain]"             # + LangChain/ChromaDB integration
pip install -e ".[llm-attribution]"       # + attention rollout (torch/transformers)
pip install -e ".[eval]"                  # + evaluation utilities (numpy, datasets)
pip install -e ".[neural-eval]"           # + optional E6/E7 sentence-transformers/FAISS/HF generator
```

---

## Synthetic Benchmark Results

> **Important caveat:** These results use template-generated adversarial documents
> evaluated directly against the policy engine (no real vector DB retrieval, no live
> LLM call, no real tool execution).  See `SOURCE_VALIDATION_REPORT.md` for scope.
> E6/E7 below add local end-to-end retrieval runs, and E8 adds sandboxed tool
> execution plus named LLM Guard/NeMo baselines. Hosted LLM/tool execution and
> additional gated guardrail baselines remain future work.

**E1: Cross-stage vs. single-stage defense (synthetic corpus, n=200 adversarial + 100 benign)**

| System | ASR ↓ | Warn-Recall ↑ | Block-Recall ↑ | Inj-Recall ↑ | blocked\_FPR ↓ |
|---|---|---|---|---|---|
| Naive content filter (proxy) | ~70% | ~30% | ~30% | ~30% | 0% |
| StruQ structural detection (proxy) | ~25% | ~75% | ~75% | ~75% | 0% |
| **RAGuard** | **~25%** | **100%** | **~75%** | **~75%** | **0%** |

Metric definitions:
- **Warn-Recall**: fraction of adversarial docs that triggered *any* violation (injection OR trust).
- **Block-Recall**: fraction of adversarial docs that were *operationally blocked* (≤ Warn-Recall).
- **Inj-Recall**: fraction where a structural injection pattern was detected.
- **blocked\_FPR**: fraction of benign queries *blocked* (trust\_risk warnings not counted).
- `warning_fpr` is 100% for RAGuard in this synthetic setup because benign external documents are LOW trust; this is alert volume, not operational blocking.

Notes:
- RAGuard Warn-Recall = 100% because memory-poisoning attacks trigger trust\_risk even without injection patterns.  Block-Recall ≈ 75% because memory-poisoning attacks are *warned* but not *blocked* (trust\_only\_blocking=False).
- Baselines are regex/keyword proxies, **not** the actual StruQ or content-filter implementations.
- ASR ~25% for RAGuard = attacks that are warned but not blocked, primarily memory-poisoning templates without structural injection cues.

**E6: End-to-end local RAG run (SQLite vector store, top-5 retrieval, n=32 attack queries + 1,000 indexed semi-real benign passages + 200 benign FPR queries)**

| Metric | Value |
|---|---:|
| Target-document retrieval hit rate | 43.75% |
| Context warning recall | 100.00% |
| Context blocking recall | 75.00% |
| Conditioned blocking recall (target in top-k) | 71.43% |
| Attack success rate | 15.62% |
| Conditioned attack success (target in top-k) | 28.57% |
| Actionable warning recall | 75.00% |
| Default-path tool attempt rate | 0.00% |
| Shadow downstream tool attempt rate | 50.00% |
| Shadow downstream tool block rate | 100.00% |
| Benign blocked FPR | 0.00% |
| Benign warning FPR | 100.00% |
| Benign actionable-warning FPR | 0.00% |
| Mean / P95 latency | 65.02 / 103.91 ms |

**E6 local baselines**

| System | Block recall | ASR | Benign blocked FPR |
|---|---:|---:|---:|
| No guard | 0.00% | 87.50% | 0.00% |
| Naive keyword filter | 59.38% | 31.25% | 0.00% |
| Structural filter | 75.00% | 15.62% | 0.00% |
| Trust-only blocking | 100.00% | 0.00% | 100.00% |
| RAGuard | 75.00% | 15.62% | 0.00% |

**E6 multi-seed summary (5 seeds, mean ± 95% CI)**

| Metric | Mean ± 95% CI |
|---|---:|
| Retrieval hit rate | 35.63% ± 6.00% |
| Context blocking recall | 70.00% ± 11.39% |
| Conditioned blocking recall | 80.95% ± 10.09% |
| Attack success rate | 15.62% ± 6.71% |
| Conditioned attack success | 19.05% ± 10.09% |
| Benign blocked FPR | 0.00% ± 0.00% |
| Benign actionable-warning FPR | 0.00% ± 0.00% |

**Small optional neural/open-weight E6 runs**

| Backend | Attack queries | Benign queries | Hit | Block | ASR |
|---|---:|---:|---:|---:|---:|
| FAISS + all-MiniLM + deterministic generation | 8 | 50 | 87.50% | 62.50% | 25.00% |
| FAISS + all-MiniLM + FLAN-T5-small generation | 8 | 50 | 87.50% | 62.50% | 25.00% |

**E7: Real-stack RAG runs (FAISS + all-MiniLM, FLAN-T5-small generation, 100 attack queries including 68 FLAN-T5 mutations, 5,000 real benign passages, 500 benign FPR queries)**

| Metric | AG News | Wikitext |
|---|---:|---:|
| Target-document retrieval hit rate | 40.00% | 43.00% |
| Context warning recall | 100.00% | 100.00% |
| Context blocking recall | 93.00% | 93.00% |
| Conditioned blocking recall (target in top-k) | 90.00% | 90.70% |
| Attack success rate | 7.00% | 7.00% |
| Conditioned attack success (target in top-k) | 10.00% | 9.30% |
| Actionable warning recall | 93.00% | 93.00% |
| Shadow downstream tool attempt rate | 13.00% | 28.00% |
| Shadow downstream tool block rate | 100.00% | 100.00% |
| Benign blocked FPR | 0.00% | 0.00% |
| Benign warning FPR | 100.00% | 100.00% |
| Benign actionable-warning FPR | 0.00% | 0.00% |
| Mean / P95 latency | 77.71 / 516.68 ms | 78.10 / 577.26 ms |

**E7 AG News generator sensitivity**

| Generator | Hit | Block | Cond. block | ASR | Mean / P95 latency |
|---|---:|---:|---:|---:|---:|
| FLAN-T5-small | 40.00% | 93.00% | 90.00% | 7.00% | 77.71 / 516.68 ms |
| SmolLM2-135M-Instruct | 40.00% | 93.00% | 90.00% | 7.00% | 239.71 / 1415.00 ms |

**E7 AG News multi-seed summary (3 seeds, mean ± 95% CI)**

| Metric | Mean ± 95% CI |
|---|---:|
| Retrieval hit rate | 41.00% ± 5.19% |
| Context blocking recall | 93.67% ± 1.31% |
| Conditioned blocking recall | 92.14% ± 5.08% |
| Attack success rate | 6.33% ± 1.31% |
| Conditioned attack success | 7.86% ± 5.08% |
| Benign blocked FPR | 0.00% ± 0.00% |
| Benign actionable-warning FPR | 0.00% ± 0.00% |
| Mean latency | 68.03 ± 12.42 ms |

**E7 AG News same-context baselines**

| System | Block recall | ASR | Benign blocked FPR |
|---|---:|---:|---:|
| No guard | 0.00% | 100.00% | 0.00% |
| Naive keyword filter | 60.00% | 40.00% | 100.00% |
| StruQ-like structural filter | 93.00% | 7.00% | 0.00% |
| External classifier (`testsavantai/prompt-injection-defender-tiny-v0`) | 73.00% | 27.00% | 0.00% |
| Trust-only blocking | 100.00% | 0.00% | 100.00% |
| RAGuard | 93.00% | 7.00% | 0.00% |

E6 is fully executable without API keys. It uses a persistent SQLite-backed vector
index with deterministic hashed lexical embeddings, a semi-real benign technical
corpus, alert triage, and a deterministic local generator by default. Optional
`sentence-transformers`, FAISS, and Hugging Face `transformers` backends are wired
in for environments that have those dependencies and local model weights. It
exercises storage, retrieval ranking, context assembly, generation, and tool
monitoring, but it is still not a commercial vector database or hosted LLM
deployment study.

E7 is the heavier reviewer-facing run. It uses a neural embedding backend,
FAISS, two real benign corpora, FLAN-T5-small generation, LLM-mutated attacks,
the SmolLM2-135M-Instruct decoder-only sensitivity run, and a working external
prompt-injection classifier baseline. In the original CPU-only E7 setting,
Qwen2.5-0.5B-Instruct and FLAN-T5-base did not complete even the small smoke
configuration; the E8 extensions add a full Qwen2.5-1.5B-Instruct run using the
accelerated causal backend. Prompt Guard / Llama Prompt Guard checkpoints were not
measured locally because the public model repositories were gated without a
Hugging Face access token. NeMo Guardrails is reproduced in a separate Python
3.11 environment with its stock YARA-backed injection-detection rail on the E8
assembled contexts. The baseline rows below are same-trace diagnostics, not
definitive comparative rankings: StruQ-like is a proxy, NeMo is untuned, and
calibration budgets are not yet standardized across all external guardrails.

**E8: Manual red-team RAG run (150 manually authored attacks including 50 regex-evasive cases, 1,000 AG News passages, 200 benign FPR queries)**

| Metric | Value |
|---|---:|
| Target-document retrieval hit rate | 82.00% |
| Context warning recall | 100.00% |
| Context blocking recall | 68.67% |
| Conditioned blocking recall (target in top-k) | 71.54% |
| Attack success rate | 0.00% |
| Conditioned attack success | 0.00% |
| External classifier blocking recall | 80.00% |
| External classifier ASR | 12.00% |
| LLM Guard blocking recall | 61.33% |
| LLM Guard ASR | 16.67% |
| LLM Guard benign blocked FPR | 0.00% |
| NeMo Guardrails blocking recall | 10.00% |
| NeMo Guardrails ASR | 14.67% |
| NeMo Guardrails benign blocked FPR | 100.00% |
| Semantic ablation blocking recall | 86.67% |
| Semantic ablation ASR | 11.33% |
| Semantic ablation benign blocked FPR | 0.00% |
| Benign blocked FPR | 0.00% |
| Benign actionable-warning FPR | 0.00% |
| Mean / P95 latency | 78.65 / 267.35 ms |

**E8 structural-visible vs. regex-evasive split**

| Attack style | n | Hit | Block | Cond. block | ASR | LLM Guard block | Semantic block |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structural-visible | 100 | 78.00% | 75.00% | 82.05% | 0.00% | 62.00% | 87.00% |
| Regex-evasive | 50 | 90.00% | 56.00% | 53.33% | 0.00% | 60.00% | 86.00% |

**E8 primary same-trace reference baselines on identical RAG contexts**

| System | Attack block | ASR | Benign blocked FPR | Warning/audit recall |
|---|---:|---:|---:|---:|
| No guard | 0.00% | 21.33% | 0.00% | -- |
| Naive keyword filter | 76.67% | 0.00% | 0.00% | -- |
| StruQ-like structural filter | 68.67% | 0.00% | 0.00% | -- |
| Trust-only blocking | 100.00% | 0.00% | 100.00% | -- |
| External prompt classifier | 80.00% | 12.00% | 0.00% | -- |
| LLM Guard PromptInjection | 61.33% | 16.67% | 0.00% | -- |
| NeMo Guardrails injection detection | 10.00% | 14.67% | 100.00% | -- |
| Semantic-intent ablation | 86.67% | 11.33% | 0.00% | -- |
| RAGuard default | 68.67% | 0.00% | 0.00% | 100.00% |

The NeMo row uses `nemoguardrails==0.21.0` and its stock
`injection_detection` action in `.venv-nemo-guardrails`. The high benign FPR is
reported as measured: the stock `sql_injection` YARA rule matches many AG News
assembled contexts, so this is a reproduced but untuned named baseline rather
than a calibrated NeMo deployment.
Rebuild the isolated baseline environment with
`python3.11 -m venv .venv-nemo-guardrails && .venv-nemo-guardrails/bin/pip install -r requirements-nemo-guardrails.txt`.

For baseline parity, future tuned comparisons should use the same calibration
budget for every guardrail: reserve a benign validation split per domain, set
thresholds or rails to the same blocked-FPR target, tune on a disjoint
development attack set with a fixed prompt/rule/label budget, and evaluate once
on held-out attacks, held-out benign passages, and the same generator/tool
traces.

**Calibration workflow**

The paper now separates calibration from evaluation: split benign corpora by
content type, freeze trust mapping and whitelist scope, sweep context/tool
thresholds on benign validation traces, choose semantic thresholds from
domain-specific ROC curves using development attacks only, and report ASR/FPR
once on held-out attacks and benign traces.

**E8 adaptive same-trace reference baselines on identical RAG contexts**

| System | Attack block | ASR | Benign blocked FPR | Warning/audit recall |
|---|---:|---:|---:|---:|
| No guard | 0.00% | 0.00% | 0.00% | -- |
| Naive keyword filter | 1.25% | 0.00% | 0.00% | -- |
| StruQ-like structural filter | 0.00% | 0.00% | 0.00% | -- |
| Trust-only blocking | 100.00% | 0.00% | 100.00% | -- |
| LLM Guard PromptInjection | 86.25% | 0.00% | 0.00% | -- |
| NeMo Guardrails injection detection | 5.00% | 0.00% | 100.00% | -- |
| Semantic-intent ablation | 97.50% | 0.00% | 0.00% | -- |
| RAGuard default | 0.00% | 0.00% | 0.00% | 100.00% |

**E8 additional validation runs**

| Run | n | Hit | Block | ASR | Benign blocked FPR | Mean / P95 latency |
|---|---:|---:|---:|---:|---:|---:|
| ChromaDB + FLAN-T5-small | 150 | 82.00% | 68.67% | 0.00% | 0.00% | 88.35 / 345.38 ms |
| FAISS + Qwen2.5-1.5B-Instruct | 150 | 82.00% | 68.67% | 0.00% | 0.00% | 1073.68 / 2820.17 ms |
| Holdout + FLAN-T5-small | 50 | 98.00% | 28.00% | 0.00% | 0.00% | 54.24 / 288.26 ms |
| Holdout + Qwen2.5-1.5B-Instruct | 50 | 98.00% | 28.00% | 0.00% | 0.00% | 526.63 / 2714.57 ms |
| Adaptive policy-aware + FLAN-T5-small | 80 | 81.25% | 0.00% | 0.00% | 0.00% | 317.84 / 1706.05 ms |

The holdout set is independent of the 150-case E8 corpus and stores two
matching category labels plus an adjudicated label per attack. It is still an
artifact-defined holdout, not an externally recruited human-red-team study.

The semantic ablation uses all-MiniLM chunk embeddings and unsafe-intent
prototype similarity at threshold 0.30. It is reported as an ablation, not as
the default blocking policy.

**E8 semantic threshold calibration**

This document-level sweep uses 50 regex-evasive attacks and 200 benign passages
per corpus. The risky-benign corpus contains legitimate security, code,
incident-response, API, and compliance text.

| Threshold | Regex-evasive block | AG News FPR | Wikitext FPR | Risky-benign FPR |
|---:|---:|---:|---:|---:|
| 0.25 | 62.00% | 9.00% | 0.50% | 100.00% |
| 0.30 | 42.00% | 4.00% | 0.00% | 91.00% |
| 0.35 | 30.00% | 1.50% | 0.00% | 71.00% |
| 0.40 | 14.00% | 0.00% | 0.00% | 56.00% |

This is why semantic blocking is not enabled by default: it recovers some
regex-evasive coverage, but risky benign operational text needs
domain-specific calibration.

**E8 semantic ROC/domain calibration**

Positives combine 50 regex-evasive and 80 adaptive attacks. Each row calibrates
against a separate 200-document benign corpus.

| Benign domain | AUC | Threshold / TPR @1% FPR | Threshold / TPR @5% FPR | Threshold / TPR @10% FPR |
|---|---:|---:|---:|---:|
| AG News | 0.8607 | 0.3881 / 19.23% | 0.2830 / 54.62% | 0.2404 / 63.08% |
| Wikitext | 0.9606 | 0.2067 / 70.77% | 0.1673 / 83.85% | 0.1434 / 88.46% |
| Risky benign | 0.1796 | 0.5348 / 0.00% | 0.5253 / 0.00% | 0.5130 / 0.00% |

The deployable policy is content-type specific: semantic hard-blocking is
enabled for news/general-web and encyclopedic/wiki sources only after domain
calibration, while security/code/compliance sources keep the same detector in
review-only mode to avoid high benign blocked FPR.

**E8 focused regex-evasive generator stress**

This run isolates the 50 regex-evasive attacks and swaps FLAN-T5-small for
SmolLM2-135M-Instruct. It shows that warning visibility survives, but hard
blocking disappears when no structural pattern evidence is present.

| Metric | Value |
|---|---:|
| Target-document retrieval hit rate | 96.00% |
| Context warning recall | 100.00% |
| Context blocking recall | 0.00% |
| Conditioned blocking recall | 0.00% |
| Semantic ablation blocking recall | 76.00% |
| Attack success rate | 0.00% |
| Generator ignored attack | 48 / 50 |
| Mean / P95 latency | 1297.70 / 2337.63 ms |

**E8 ASR outcome decomposition**

| Primary outcome | Count | Rate |
|---|---:|---:|
| Target not retrieved in top-k | 27 | 18.00% |
| Blocked by context monitor | 88 | 58.67% |
| Blocked by tool monitor | 0 | 0.00% |
| Retrieved but generator ignored attack | 35 | 23.33% |
| Unsafe generation without tool | 0 | 0.00% |
| Executed successfully | 0 | 0.00% |

**E8: Real local plugin/tool workflow**

| Metric | Value |
|---|---:|
| Malicious tool calls | 8 |
| Benign tool calls | 8 |
| Malicious block rate | 100.00% |
| Malicious execution rate | 0.00% |
| Benign pass/execution rate | 100.00% |
| Benign false-block rate | 0.00% |
| Reviewed benign low-trust calls | 2 |
| Blocked malicious side effects | 8 / 8 |

**E9: Calibrated reviewer-improvement studies**

The E9 artifacts are saved in `output/experiments/e9_*.json`. They include
dependency-light calibration traces for reproducibility and a full neural
FAISS/all-MiniLM replay with FLAN-T5-small and Qwen2.5-1.5B-Instruct.

| Study | Key result |
|---|---|
| Hybrid calibrated defense | Structural-or-semantic hybrid: 81.74% held-out block recall, 0.00% ASR, 0.00% benign blocked FPR |
| Full neural hybrid replay | FAISS + sentence-transformers over the same 150 manual + 80 adaptive attacks: hybrid reaches 100.00% block recall, 0.00% ASR, 0.00% FPR on AG News/Wikitext for both FLAN-T5-small and Qwen2.5-1.5B; risky-benign removes ASR with 80.00% benign blocked FPR |
| Tuned baseline parity | Tuned semantic: 81.74% block recall at 0.00% benign blocked FPR; untuned keyword any has 100.00% benign blocked FPR |
| Aggregation ablation | Max-taint: 100.00% attack review/block recall but 34.33% benign review FPR; attribution-weighted: 90.83% attack review recall and 0.00% benign review FPR |
| Scaled operator cost | 5,000 trust-only query events aggregate to 500 source/day events; 0 actionable alerts per 1,000 benign queries; 12.50% benign tool review rate |

**Operator triage cost**

Trust-only benign warnings remain audit events, not paging alerts. In the E6,
E7, E8 red-team, and E8 adaptive RAG runs, benign actionable alerts are 0 and
benign blocks are 0 despite 100% trust-only informational warning volume. In
the plugin workflow, 2/8 benign low-trust privileged calls enter review and 0/8
are blocked. The intended mitigation is source-level aggregation by signed
manifest/domain/policy version, allowlisting for attested sources, sampling of
repeated informational events, and bounded review queues for low-trust tool use.
Recommended reporting separates informational audit events per source/day,
actionable alerts per 1,000 queries, and tool-review items per privileged tool
call.

**Authenticated provenance integration**

The default implementation assigns trust from local metadata and SHA-256 hashes,
which provide integrity fingerprints but not authenticated source identity. The
prototype now includes an optional local HMAC-signed provenance mode:
`provenance_signing_key` signs source/chunk manifests at ingestion, and
`require_signed_high_trust=True` makes retrieval degrade unsigned HIGH/SYSTEM or
whitelisted sources to LOW trust. This mitigates local metadata spoofing, but
full asymmetric publisher attestation, key rotation, revocation, and freshness
checks remain production work.

**Aggregation and whitelist caveats**

RAGuard uses max-taint aggregation so low-trust content cannot be diluted by
nearby trusted text. This is conservative and can over-taint mixed contexts, so
the paper now calls out unevaluated alternatives: retrieval-score-weighted,
length-weighted, and attribution-weighted taint. Whitelist bypasses should be
scoped, signed, expiring, audited, and should never suppress structural
injection scanning or the untrusted fraction from other chunks.
The proposed ablation metric is review-volume reduction at fixed ASR and fixed
benign blocked FPR, so aggregation changes are judged by reliability tradeoff
rather than by convenience.

This workflow executes actual sandboxed tools rather than mocks: localhost HTTP
fetch, SQLite CRM export/update, filesystem search, report writing, and JSONL
outbox write. Malicious tainted calls use the same tools and are blocked before
execution.

**E3: Attribution accuracy (multi-source context, 4 distractors per query)**

| Method | Attribution accuracy |
|---|---|
| RAGuard n-gram overlap | ~100% (**inflated** — see caveat) |
| Random baseline | ~25% (1 of 5 sources, chance level) |

**E3 caveat:** The simulated LLM output is a literal prefix of the adversarial
document, so n-gram overlap attribution identifies the correct source almost
trivially.  This value should **not** be cited as a real attribution claim.
Real attribution accuracy on LLM-generated paraphrases requires open-weights
model access and human-annotated ground truth.
See `SOURCE_VALIDATION_REPORT.md` §2.4.

**Manuscript status after reviewer-style tightening:** The paper now frames E1-E5
as controlled component validation, E6 as an end-to-end local artifact run, E7
as the heavier neural-retrieval/real-corpus run, and E8 as a 150-case
manual-red-team plus regex-evasive/sandboxed-tool stress test. The latest E8
additions include a ChromaDB backend, Qwen2.5-1.5B-Instruct generation, and a
50-case annotated holdout. A venue-ready reliability study still needs larger
hosted or GPU-scale open-weight LLM result sets, tuned external guardrail
deployments, production plugin execution, and independent external
human-red-team annotation.

---

## LangChain Integration

```python
from langchain.vectorstores import Chroma
from ragentguard.pipeline import LangChainRetrieverWrapper
from ragentguard import RAGentGuard, RAGentGuardConfig

retriever = Chroma(...).as_retriever()
guard = RAGentGuard(RAGentGuardConfig.default())
taint_retriever = LangChainRetrieverWrapper(retriever, guard.config)

context, taint, chunks = taint_retriever.assemble_context(query)
guard.check_context(context, chunks, query)
# ... LLM call ...
guard.check_tool_call("tool_name", args, taint)
```

---

## Citing

```bibtex
@article{ragentguard2025,
  title  = {RAGuard: Cross-Stage Reliability Auditing for
             Retrieval-Augmented Generation Agents},
  author = {},
  year   = {2025},
  note   = {Preprint}
}
```
