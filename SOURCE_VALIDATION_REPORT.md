# RAGuard — Source Validation Report

This document enumerates exactly what is **fully implemented**, what is **simulated / proxied**,
and what **requires real infrastructure** before results can be submitted to a top-tier venue.

---

## 1. Fully Implemented (no external dependencies required)

| Component | File | Notes |
|---|---|---|
| `ProvenanceTag` data structure | `core/provenance.py` | Full serialization/deserialization |
| `TaintVector` with merge, advance\_stage | `core/provenance.py` | Immutable propagation semantics |
| `PolicyViolation` record | `core/provenance.py` | JSON-serialisable |
| Two-dimensional policy model | `core/policy.py` | injection\_risk (blocking) + trust\_risk (warning) |
| 7 structural injection pattern families | `core/policy.py` | StruQ-inspired regex patterns, including attacker URL/e-mail sinks |
| `RAGentGuardConfig` with presets | `core/config.py` | default / strict / research\_mode |
| Stage 1: DocumentIngestor | `pipeline/ingestion.py` | SHA-256 source ID, sliding-window chunking, trust heuristics |
| Stage 2: TaintPropagator | `pipeline/retrieval.py` | Taint score from trust level |
| Stage 2: LangChainRetrieverWrapper | `pipeline/retrieval.py` | Adapter for LangChain BaseRetriever |
| Stage 3: ContextPolicyMonitor | `monitors/context_monitor.py` | Pre-LLM gate with PolicyBlockedError |
| Stage 4: GenerationAttributor (overlap) | `pipeline/attribution.py` | n-gram overlap, no model required |
| Stage 4: GenerationAttributor (attention rollout) | `pipeline/attribution.py` | **Implemented** but requires `torch` + `transformers` |
| Stage 4: GenerationAttributor (gradient) | `pipeline/attribution.py` | **Implemented** but requires `torch` |
| Stage 5: ToolCallMonitor | `monitors/tool_monitor.py` | @guard decorator, ToolCallBlockedError |
| Stage 6: AuditReporter (JSON + HTML) | `audit/reporter.py` | Per-session report with mitigations |
| Stage 6: AuditLogger (JSONL) | `audit/reporter.py` | Real-time streaming log |
| RAGuard orchestrator | `ragentguard.py` | Wires all 7 pipeline stages |
| Adversarial corpus generator | `attacks/corpus.py` | 4 categories × N docs via templates |
| Evaluation metrics | `evaluation/metrics.py` | ASR, warning/blocking/injection recall, blocked\_fpr/warning\_fpr, Attribution%, Latency P95 |
| E1–E5 experiment runners | `evaluation/experiments.py` | See scope caveats below |
| E6 end-to-end local RAG runner | `evaluation/end_to_end.py` | SQLite vector store, ranked retrieval, semi-real benign corpus, triage, local baselines |
| Optional E6 neural/open-weight backends | `evaluation/end_to_end.py` | sentence-transformers, FAISS, transformers if installed |
| E7 real-stack RAG runner | `evaluation/end_to_end.py` | FAISS + sentence-transformers, FLAN-T5-small generation, SmolLM2 sensitivity, AG News/Wikitext corpora, 68 mutated attacks |
| E8 manual red-team/tool runner | `evaluation/end_to_end.py` | 150 manually authored attacks, including 50 regex-evasive cases, plus ASR outcome decomposition and sandboxed local tool execution |
| E8 Chroma/Qwen/holdout extensions | `evaluation/end_to_end.py` | ChromaDB vector store, Qwen2.5-1.5B-Instruct generation, and a 50-case annotated holdout red-team set |
| 28 pytest cases (25 test functions) | `tests/test_core.py` | Include FPR regression, metric-dimension tests, and E6/E8 smoke tests |

---

## 2. Simulated / Proxied (results are synthetic, not from real RAG deployments)

### 2.1 Adversarial Corpus

**What is done:** Template-based generation of 500 adversarial documents (4 categories × 125).

**Caveat:** Templates are hand-authored; they do not cover the full distribution of real
adversarial prompts.  Real red-team evaluation requires human adversaries crafting
novel injections against a live RAG system.

**Gap for paper:** Replace templates with a human-red-teamed corpus validated against
at least one state-of-the-art language model.  Target: ≥500 docs with inter-annotator agreement
on attack category labels.

---

### 2.2 Baseline Comparisons (E1)

**What is done:** Two proxy baselines:
- `naive_filter`: keyword list matching
- `struq_proxy`: applies RAGuard's injection pattern scanner only

**Caveat:** Neither is the actual implementation of StruQ or a published content filter.
The `struq_proxy` uses the **same** regex patterns as RAGuard Stage 3, making the
comparison partially tautological for the injection pattern dimension.

**Gap for paper:** Either (a) run the actual StruQ implementation from `../StruQ/` on the
same corpus and RAG pipeline, or (b) label comparisons explicitly as "proxy baselines"
and limit claims accordingly.

---

### 2.3 Detection Evaluation (E1, E2, E5)

**What is done:** `_evaluate_with_ragentguard()` passes a document's raw text directly
to `PolicyEngine.assess_context()`.  This is a **single-stage text evaluation**, not a
full RAG pipeline evaluation.

**What is missing in E1/E2/E5:**
1. No standard vector DB retrieval (Chroma / Weaviate / Pinecone)
2. No neural embedding model
3. No LLM generation
4. No real tool execution chain
5. Multi-doc context mixing: adversarial doc evaluated alone, not alongside benign distractors

**Gap for paper:** E6 now closes this at artifact level with a local SQLite vector
index, ranked retrieval, multi-document contexts, deterministic generation, and safe
simulated tool dispatch. A production-grade paper should still wire up a standard
LangChain + ChromaDB pipeline.  Ingest both benign (BEIR / Wikipedia) and
adversarial documents.  Run real retrieval queries.  Measure detection on the
assembled multi-doc context after real retrieval.

---

### 2.3b End-to-End Local RAG Evaluation (E6)

**What is done:** `run_e6_end_to_end_rag()` exercises ingestion, embedding,
persistent vector storage, cosine-ranked top-k retrieval, multi-document context
assembly, context monitoring, generation, tool-call monitoring, and safe
simulated tool execution. The default backend uses deterministic hashed lexical
embeddings, SQLite vector storage, a semi-real benign technical corpus, and a
deterministic local generator. Optional sentence-transformer embeddings, FAISS
retrieval, and Hugging Face `transformers` generation are wired in when those
packages and model weights are installed locally.

**Measured default artifact run:** 32 attack queries, 1,000 indexed semi-real
benign passages, 200 benign FPR queries, top-5 retrieval. Target-document
retrieval hit rate 43.75%, context warning recall 100%, context blocking recall
75.00%, conditioned blocking recall 71.43% when the adversarial target appears
in top-k retrieval, attack success rate 15.62%, conditioned attack success
28.57%, benign blocked FPR 0%, benign warning FPR 100%, benign actionable-warning
FPR 0%, mean/P95 latency 65.02/103.91 ms. The
default path attempts no tools because the context gate blocks risky contexts
first; a shadow downstream pass shows 50.00% tool-attempt rate and 100%
tool-block rate if blocked contexts are allowed to continue to generation.

**Five-seed summary:** retrieval hit rate 35.63% ± 6.00%, context blocking
recall 70.00% ± 11.39%, conditioned blocking recall 80.95% ± 10.09%, attack
success rate 15.62% ± 6.71%, conditioned attack success 19.05% ± 10.09%, benign
blocked FPR 0% ± 0%, and benign actionable-warning FPR 0% ± 0%.

**Local baselines:** no guard (ASR 87.50%), naive keyword filter (block recall
59.38%, ASR 31.25%), structural filter (block recall 75.00%, ASR 15.62%), and
trust-only blocking (block recall 100%, ASR 0%, benign blocked FPR 100%).

**Small optional neural/open-weight runs:** FAISS + all-MiniLM embeddings with
deterministic generation gives 87.50% retrieval hit rate, 62.50% block recall,
and 25.00% ASR over 8 attack queries and 50 benign FPR queries. Replacing the
generator with FLAN-T5-small gives the same safety metrics on this small run,
with higher latency.

**Caveat:** E6 is an executable end-to-end artifact test, but it is still not a
production deployment. The default measured run uses a local SQLite dense-vector
index rather than Chroma/Weaviate/Pinecone, hashed lexical embeddings rather
than neural embeddings, and deterministic generation rather than a hosted or
open-weight LLM. The optional neural/open-weight runs are intentionally small
feasibility checks, not full neural benchmarks.

**Gap for paper:** E7 now covers neural retrieval, two larger real benign
corpora, and a full 100-query FLAN-T5-small generator run; E8 adds ChromaDB,
Qwen2.5-1.5B-Instruct, and sandboxed local tools. Remaining work is a full
LangChain-style app integration, production plugins, and hosted or GPU-scale
larger open-weight generation.

---

### 2.3c Real-Stack RAG Evaluation (E7)

**What is done:** `run_e7_real_stack()` is a reproducible heavier configuration
of the end-to-end harness. It uses FAISS, `sentence-transformers/all-MiniLM-L6-v2`,
FLAN-T5-small generation, 5,000-passages from AG News or Wikitext as the benign
corpus, the original 32 synthetic attack queries, and 68 FLAN-T5-mutated attack
documents for 100 attack queries total. It also supports a Hugging Face
text-classification guard baseline on the same retrieved contexts.

**Measured AG News run:** 100 attack queries, 500 benign FPR queries, top-5
retrieval. Target-document retrieval hit rate 40.00%, context warning recall
100%, context blocking recall 93.00%, conditioned blocking recall 90.00% when
the adversarial target appears in top-k retrieval, attack success rate 7.00%,
conditioned attack success 10.00%, benign blocked FPR 0%, benign warning FPR
100%, benign actionable-warning FPR 0%, and mean/P95 latency 77.71/516.68 ms.
The shadow downstream pass shows 13.00% tool-attempt rate and 100% tool-block
rate if blocked contexts are allowed to continue to generation.

**Measured Wikitext run:** same configuration with Wikitext gives 43.00%
retrieval hit rate, 93.00% context blocking recall, 90.70% conditioned blocking
recall, 7.00% attack success, 9.30% conditioned attack success, 0% benign
blocked FPR, 0% benign actionable-warning FPR, and 78.10/577.26 ms mean/P95
latency.

**Generator sensitivity:** Replacing FLAN-T5-small with
`HuggingFaceTB/SmolLM2-135M-Instruct`, a decoder-only instruction model, on the
AG News run preserves the safety metrics: 40.00% retrieval hit rate, 93.00%
context blocking recall, 90.00% conditioned blocking recall, 7.00% attack
success, and 0% benign actionable-warning FPR. Latency increases to
239.71/1415.00 ms mean/P95. In the original CPU-only E7 setting,
`Qwen/Qwen2.5-0.5B-Instruct` and `google/flan-t5-base` did not complete the
small smoke configuration; E8 later adds a full Qwen2.5-1.5B-Instruct run using
the accelerated causal backend.

**Three-seed AG News summary:** retrieval hit rate 41.00% ± 5.19%, context
blocking recall 93.67% ± 1.31%, conditioned blocking recall 92.14% ± 5.08%,
attack success rate 6.33% ± 1.31%, conditioned attack success 7.86% ± 5.08%,
benign blocked FPR 0% ± 0%, benign actionable-warning FPR 0% ± 0%, and mean
latency 68.03 ± 12.42 ms.

**Same-context baselines on AG News:** no guard (ASR 100%), naive keyword filter
(block recall 60.00%, ASR 40.00%, benign blocked FPR 100%), StruQ-like
structural filter (block recall 93.00%, ASR 7.00%, benign blocked FPR 0%),
external classifier `testsavantai/prompt-injection-defender-tiny-v0` (block
recall 73.00%, ASR 27.00%, benign blocked FPR 0%), and trust-only blocking
(block recall 100%, ASR 0%, benign blocked FPR 100%).

**External baseline access:** `testsavantai/prompt-injection-defender-tiny-v0`
was downloaded and measured as a working external classifier baseline. Prompt
Guard and Llama Prompt Guard were not measured because the relevant Hugging Face
model repositories were gated without a local access token. NeMo Guardrails
could not be used in the local Python 3.14 neural environment through a
LangChain/Pydantic compatibility path, but its stock injection-detection rail is
now reproduced in a separate Python 3.11 environment and reported in E8.

**Caveat:** E7 is a local neural-retrieval artifact run, not a hosted production
RAG audit. It does not execute real external tools.

---

### 2.3d Manual Red-Team and Sandboxed Tool Evaluation (E8)

**What is done:** `human_redteam_attack_set()` defines 150 manually authored
attacks that are independent of the template generator: 100 structurally
visible cases and 50 regex-evasive cases with zero matches against the seven
current structural pattern families. The documents use different domains,
wording, and formatting from the synthetic templates.
`run_e8_human_redteam_rag()` evaluates them through the same
FAISS/all-MiniLM/FLAN-T5-small path with a 1,000-passage AG News benign corpus
and 200 benign FPR queries, and records a primary ASR outcome label for each
attack. `run_e8_sandboxed_tool_execution()` executes harmless local tool stubs
through the real `ToolCallMonitor`: calculator, file search in a temporary
directory, mock HTTP fetch, and mock e-mail send.
`run_e8_real_plugin_tool_workflow()` goes beyond stubs by executing a concrete
local plugin workflow: HTTP fetches against a temporary localhost server, SQLite
CRM export/update, filesystem search, report writing, and JSONL outbox writes.

**Manual red-team run:** 150 attack queries, 200 benign FPR queries, top-5
retrieval. Target-document retrieval hit rate 82.00%, context warning recall
100%, context blocking recall 68.67%, conditioned blocking recall 71.54%, attack
success rate 0%, conditioned attack success 0%, benign blocked FPR 0%, benign
actionable-warning FPR 0%, and mean/P95 latency 78.65/267.35 ms. The
structural-visible slice (n=100) has 78.00% retrieval hit, 75.00% context
blocking recall, 82.05% conditioned blocking recall, and 0% ASR. The
regex-evasive slice (n=50) has 90.00% retrieval hit, 56.00% context blocking
recall, 53.33% conditioned blocking recall, and 0% ASR. The semantic-intent
ablation uses all-MiniLM chunk embeddings and unsafe-intent prototype
similarity at threshold 0.30. It blocks 86.67% of all attacks, 87.00% of
structural-visible attacks, and 86.00% of regex-evasive attacks, with 11.33%
ablation ASR and 0% benign blocked FPR on the E8 AG News FPR queries. A
separate document-level threshold sweep shows that this threshold is not
production-calibrated: on 50 regex-evasive attacks and 200 benign documents per
corpus, thresholds 0.25/0.30/0.35/0.40 give regex-evasive block rates of
62.00%/42.00%/30.00%/14.00%, AG News FPR of 9.00%/4.00%/1.50%/0.00%,
Wikitext FPR of 0.50%/0.00%/0.00%/0.00%, and risky-benign FPR of
100.00%/91.00%/71.00%/56.00%. The risky-benign set contains legitimate
security, code, incident-response, API, and compliance passages.
`run_e8_semantic_roc_analysis()` adds domain-calibrated ROC analysis using the
combined 130 regex-evasive/adaptive attacks as positives. AUC is 0.8607 against
AG News, 0.9606 against Wikitext, and 0.1796 against risky-benign text. At a
5% benign blocked-FPR budget, calibrated thresholds/TPR are 0.2830/54.62% for
AG News, 0.1673/83.85% for Wikitext, and 0.5253/0.00% for risky-benign text.
The resulting deployable policy is content-type specific: semantic hard-blocking
is justified only for calibrated news/general-web and encyclopedic/wiki domains,
while risky-benign security/code/compliance content keeps the semantic score in
review-only mode to avoid high benign blocked FPR.
The primary ASR outcome
decomposition is: 27 target documents not retrieved, 88 attacks blocked by the
context monitor, 0 blocked by the tool monitor, 35 retrieved
attacks ignored by the generator, 0 unsafe generations without tools, and 0
successful executions. The working external classifier baseline blocks 80.00%
with 12.00% ASR and 0% benign blocked
FPR on the same contexts.

**Focused regex-evasive generator stress:** `run_e8_regex_evasive_generator_stress()`
isolates the 50 regex-evasive attacks and runs them with the same
FAISS/all-MiniLM retrieval path plus `HuggingFaceTB/SmolLM2-135M-Instruct`
generation. This run retrieves 48/50 targets (96.00%), preserves 100% warning
recall, but blocks 0% of contexts because the target contexts contain no
structural pattern matches. ASR remains 0%: 2 cases are retrieval misses and 48
are categorized as generator-ignored attacks. Mean/P95 latency is
1297.70/2337.63 ms. The same semantic-intent ablation blocks 76.00% of this
isolated regex-evasive slice with 0% ablation ASR. This is the clearest current
evidence that regex-evasive zero-ASR results should not be interpreted as
default hard-blocking coverage, while semantic evidence can recover much of the
gap as an ablation.

**Same-trace reference baselines on identical RAG contexts:** The primary E8
comparison now aggregates all baseline rows on the same FAISS/all-MiniLM/FLAN-T5
retrieved contexts and the same 200 benign FPR contexts, but these rows should
not be read as definitive comparative rankings because several are proxies or
untuned stock deployments:
no guard (0.00% block, 21.33% ASR, 0.00% benign blocked FPR), naive keyword
filter (76.67%, 0.00%, 0.00%), StruQ-like structural filter (68.67%, 0.00%,
0.00%), trust-only blocking (100.00%, 0.00%, 100.00%), external prompt
classifier (80.00%, 12.00%, 0.00%), LLM Guard PromptInjection (61.33%,
16.67%, 0.00%), NeMo Guardrails injection detection (10.00%, 14.67%,
100.00%), semantic-intent ablation (86.67%, 11.33%, 0.00%), and RAGuard
default (68.67%, 0.00%, 0.00%, with 100.00% warning recall).

The adaptive E8 comparison uses the same 80 adaptive attack contexts and 200
benign FPR contexts for every baseline: no guard (0.00% block, 0.00% ASR,
0.00% benign blocked FPR), naive keyword filter (1.25%, 0.00%, 0.00%),
StruQ-like structural filter (0.00%, 0.00%, 0.00%), trust-only blocking
(100.00%, 0.00%, 100.00%), LLM Guard PromptInjection (86.25%, 0.00%, 0.00%),
NeMo Guardrails injection detection (5.00%, 0.00%, 100.00%), semantic-intent
ablation (97.50%, 0.00%, 0.00%), and RAGuard default (0.00%, 0.00%, 0.00%,
with 100.00% warning recall). The machine-readable aggregate is saved at
`output/experiments/e8_head_to_head_baselines.json`.

**Baseline parity caveat:** A definitive comparison still requires a shared
calibration budget across baselines: domain-specific benign validation splits,
fixed blocked-FPR targets, disjoint attack-development sets with equal
prompt/rule/label budgets, held-out attack and benign evaluation, confidence
intervals over seeds, and the same generator/tool traces. The current artifact
does not yet standardize that budget across LLM Guard, NeMo, Prompt Guard/Llama
Guard-style classifiers, StruQ, semantic detectors, and RAGuard.

**Calibration workflow:** The manuscript now makes calibration explicit:
separate benign validation/test splits by content type, freeze trust mapping and
whitelist scope, sweep context/tool thresholds only on benign validation traces,
select semantic thresholds from domain-specific ROC curves using development
attacks, then report ASR, blocked FPR, actionable FPR, and review load once on
held-out attacks and benign traces.

**Named guardrail baseline:** `scripts/llm_guard_baseline.py` runs the actual
LLM Guard `PromptInjection` scanner in a dedicated Python 3.11 environment
(`.venv-llm-guard`) against the exported E8 assembled RAG contexts. With
threshold 0.9 and the public `testsavantai/prompt-injection-defender-tiny-v0`
classification model, it blocks 61.33% of red-team attack contexts, leaves
16.67% ASR, blocks 0% of benign contexts, and averages 14.30 ms per scan. It
blocks 62.00% of the structural-visible slice and 60.00% of the regex-evasive
slice. On the adaptive E8 contexts, it blocks 86.25% of attacks, leaves 0.00%
ASR, blocks 0.00% of benign contexts, and averages 12.50 ms per scan. The
default ProtectAI model path was not used because the available local access
constraints made the public TestsavantAI model the reproducible option.

`scripts/nemo_guardrails_baseline.py` runs NeMo Guardrails 0.21.0
`injection_detection` with the stock code/SQL/template/XSS YARA rules in a
separate Python 3.11 environment (`.venv-nemo-guardrails`). On the same E8
assembled contexts, it blocks 10.00% of attacks, leaves 14.67% ASR, blocks
100.00% of benign AG News contexts, and averages 1.09 ms per scan. The slice
blocking recalls are 9.00% for structural-visible attacks and 12.00% for
regex-evasive attacks. All 215 detections are from the stock `sql_injection`
rule. On the adaptive contexts, it blocks 5.00% of attacks, leaves 0.00% ASR,
blocks 100.00% of benign contexts, and averages 1.42 ms per scan. This is
recorded as a reproduced but untuned named baseline rather than a calibrated
NeMo deployment.

**Sandboxed tool run:** 4 malicious calls and 5 benign calls. Malicious block
rate is 100%, malicious execution rate is 0%, benign pass/execution rate is
100%, benign false-block rate is 0%, and one benign low-trust call is reviewed
but not blocked. The experiment performs actual local function calls for benign
tools, but no shell, network, or external e-mail side effects occur.

**Additional E8 validation:** `run_e8_chromadb_rag()` replaces FAISS with
ChromaDB on the same 150 attack queries and reproduces the primary metrics:
82.00% retrieval hit, 68.67% context blocking recall, 71.54% conditioned
blocking recall, 0% ASR, 0% benign blocked FPR, and 88.35/345.38 ms mean/P95
latency. `run_e8_stronger_generator_rag()` runs the same 150-case set with
Qwen2.5-1.5B-Instruct on the causal-transformer backend. It preserves 82.00%
retrieval hit, 68.67% context blocking recall, 71.54% conditioned blocking
recall, 0% ASR, and 0% benign blocked FPR, with 1073.68/2820.17 ms mean/P95
latency. `run_e8_independent_annotated_redteam_rag()` adds a 50-case annotated
holdout set independent of the 150-case corpus. The holdout has 98.00%
retrieval hit, 28.00% context blocking recall, 28.57% conditioned blocking
recall, 0% ASR, and 0% benign blocked FPR with FLAN-T5-small; Qwen2.5-1.5B
gives the same safety metrics with 526.63/2714.57 ms mean/P95 latency.

`run_e8_adaptive_redteam_rag()` adds 80 policy-aware adaptive attacks with zero
matches against the structural pattern families. It records 81.25% retrieval
hit, 100% warning recall, 0% context blocking recall, 0% ASR, and 0% benign
blocked FPR; 15 attacks are not retrieved and 65 retrieved attacks are ignored
by FLAN-T5-small. The semantic-intent ablation blocks 97.50% of these adaptive
contexts at 0% AG News benign blocked FPR, but remains an ablation because the
risky-benign calibration above shows high domain sensitivity.

**E9 calibrated reviewer-improvement studies:** Five runners address the
remaining reviewer concerns. `run_e9_hybrid_calibrated_defense()` tunes the
semantic threshold on a development split at a 5% benign blocked-FPR budget and
evaluates a held-out split; the structural-or-semantic hybrid reaches 81.74%
blocking recall, 0.00% ASR, and 0.00% benign blocked FPR in the lightweight
local trace. `run_e9_full_neural_hybrid_defense()` replays the same 150 manual
red-team plus 80 adaptive attacks through FAISS, all-MiniLM sentence-transformer
embeddings, and FLAN-T5-small or Qwen2.5-1.5B-Instruct. On AG News and Wikitext,
the calibrated hybrid reaches 100.00% blocking recall, 0.00% ASR, and 0.00%
benign blocked FPR for both generators; on risky-benign security/code/compliance
queries, it removes ASR only with 80.00% benign blocked FPR, so semantic blocking
remains content-type calibrated rather than a universal default. `run_e9_tuned_baseline_parity()` applies the same dev/test split
and FPR budget to dependency-light baselines: tuned semantic reaches 81.74%
blocking recall at 0.00% benign blocked FPR, whereas untuned keyword-any blocks
24.35% of attacks but also blocks 100.00% of benign traces. `run_e9_aggregation_ablation()`
compares max-taint, length-weighted, retrieval-score-weighted,
attribution-weighted, and hybrid aggregation on 120 malicious and 300 benign
mixed contexts. Max-taint preserves 100.00% attack review/block recall but
34.33% benign review/block FPR; attribution-weighted taint preserves 90.83%
attack review recall with 0.00% benign review FPR. `run_e9_scaled_operator_cost_study()`
scales benign query cost to 5,000 trust-only events over 500 source clusters,
reducing query-level informational events by 90.00% after source/day
aggregation, with 0 actionable alerts or benign blocks per 1,000 benign
queries; 100 malicious tool calls are blocked and 200 benign tool calls have
0.00% false blocks and 12.50% review rate.

**Real local plugin/tool workflow:** 8 malicious calls and 8 benign calls use
the same local plugin interfaces: `http_get`, `crm_export`,
`crm_update_status`, `file_search`, `write_report`, and `send_email`. All 8
malicious calls are blocked before execution and produce no sandbox side-effect
changes. All 8 benign calls execute, including actual localhost HTTP fetch,
SQLite export/update, file search, report write, and outbox write; two benign
low-trust calls are reviewed but not blocked.

**Operator triage cost and mitigations:** Trust-only warnings are deliberately
non-paging audit events. E6 produces 200/200 benign trust-only informational
events, E7 produces 500/500, and the E8 red-team/adaptive RAG runs each produce
200/200, but all four have 0 benign actionable alerts and 0 benign blocks after
triage. The plugin workflow sends 2/8 benign low-trust privileged calls to
human review and blocks 0/8 benign calls. The intended deployment mitigations
are aggregation by signed source manifest, domain, and policy version;
allowlisting or suppression for authenticated sources; sampling repeated
trust-only events; reserving paging/blocking for structural or calibrated
semantic injection evidence; and bounded review queues for low-trust tool
actions.
Recommended cost reporting separates informational audit events per source/day,
actionable alerts per 1,000 queries, and human-review items per privileged tool
call, because these are different operational burdens.

**Authenticated provenance integration:** The default artifact uses local
metadata/path trust assignments and SHA-256 hashes. These hashes are integrity
fingerprints, not authenticated source identity. The implementation now includes
an optional local HMAC-signed manifest path: `provenance_signing_key` signs
source/chunk manifests at ingestion, and `require_signed_high_trust=True`
degrades unsigned HIGH/SYSTEM or whitelisted sources to LOW trust at retrieval.
This mitigates local metadata spoofing when the signing key is protected, but it
is not full asymmetric publisher attestation with key rotation, revocation, or
freshness checks.

**Aggregation and whitelist caveats:** Max-taint aggregation preserves
worst-source visibility but can over-taint mixed contexts. The paper now
identifies retrieval-score-weighted, length-weighted, and attribution-weighted
taint as missing ablations. Whitelist bypasses should be scoped, signed,
expiring, audited, and should never suppress structural injection scanning or
the untrusted fraction contributed by non-whitelisted chunks.
The proposed ablation target is review-volume reduction at fixed ASR and fixed
benign blocked FPR.

**Caveat:** The 150-case red-team set and the 50-case annotated holdout are
manually authored within the artifact. The holdout records duplicate labels and
an adjudicated label for reproducibility, but it is not an externally recruited
or independently staffed human-red-team study. The real plugin workflow uses
local sandboxed plugins, not production SaaS plugins.

---

### 2.4 Attribution Experiment (E3)

**What is done:** Multi-source context (1 adversarial + 4 benign distractor chunks).
Simulated output reuses the first 128 characters of the adversarial document.

**Caveat:** Because the "generated output" is literally a prefix of the adversarial doc,
n-gram overlap attribution achieves near-100% accuracy trivially.  This does not measure
attribution quality on LLM-generated paraphrases of adversarial content.

**Gap for paper:**
1. Generate real language-model output using a production-grade model with the multi-source context as input.
2. Annotate which source(s) influenced which output spans (human ground truth).
3. Compare overlap, attention rollout (open-weights model), and gradient attribution against
   the TracLLM baseline on the same corpus.

---

### 2.5 FPR Measurement

**What is done:** FPR = fraction of benign documents (LOW trust) that are **operationally
blocked** by the policy engine.  With `trust_only_blocking=False`, benign documents with
no injection patterns are never blocked → FPR = 0%.

**Caveat:** The benign "documents" are short synthetic strings.  Real benign corpora
(Wikipedia, code documentation, technical PDFs) may contain strings that incidentally
match injection patterns (e.g., code snippets with backticks, configuration files with
`always` directives, educational content about prompt injection).

**Gap for paper:** Measure FPR on a real benign corpus (BEIR MS-MARCO or Wikipedia
passages with 10,000 documents) to quantify the natural false-positive rate of the 7
injection patterns on non-adversarial text.

---

### 2.6 Latency Overhead (E4)

**What is done:** Measures Python function call time of `PolicyEngine.assess_context()`
(regex matching on text, no I/O).  Result: ~0.05 ms per call.

**Caveat:** Real latency overhead includes:
- Provenance tag lookup from vector DB metadata
- Attribution model inference (attention rollout: ~10–100 ms on GPU)
- Extra disk I/O for audit log append

**Gap for paper:** Measure end-to-end latency on a full LangChain application
with ChromaDB retrieval and a production-style generator, then report the
latency ratio (with vs. without RAGuard) across N queries.

---

## 3. Requires Real Infrastructure (manual execution)

These steps cannot be automated within this code artifact and require external resources:

| Task | Requirement | Priority for paper |
|---|---|---|
| LangChain agent integration test | Full LangChain app wiring plus API key or local LLM | High |
| Hosted/larger open-weight generation run | API key or local GPU for stronger 50-100+ full-stack generations | Critical |
| Llama Guard / Prompt Guard baselines | Hugging Face gated-model access token or alternate release | Critical |
| Tuned NeMo Guardrails deployment | Calibrated rails/rules for full RAG contexts | High |
| LLM-generated attribution ground truth | LLM API key (provider of choice) | High |
| Attention rollout attribution (open weights) | GPU + `pip install torch transformers` + open-weights model | High |
| Independent human red-team corpus validation | Manual annotation by 2+ external security experts | High |
| Real StruQ baseline | Run `../StruQ/` implementation on same corpus | Medium |
| FPR on additional real benign corpora | Download Wikipedia / MS-MARCO passages | Medium |
| E5 real-world app audit | API access to 3+ commercial RAG application plugins | Medium |

---

## 4. Known Limitations of the Policy Engine

| Limitation | Impact | Proposed fix |
|---|---|---|
| Memory-poisoning attacks have no structural patterns → not blocked (only warned) | ASR ~100% for this category in E2 | Add semantic anomaly detection (embedding drift, topic shift) |
| Cross-tool-taint templates are now covered in E2, but coverage is lexical | ASR 0% for current templates; paraphrases may evade | Add semantic parser and real tool sandbox evaluation |
| Pattern scanner is purely lexical (regex) | Adversarial paraphrasing evades detection | Add model-based secondary classifier |
| Trust level assigned heuristically from URL/path by default | Attacker can spoof a trusted-looking source path unless signed mode is enabled | Use optional HMAC manifests now; require asymmetric attestation for production HIGH trust |
| Max-taint aggregation is conservative | Mixed contexts may be over-tainted and routed to review | Evaluate retrieval-score-, length-, and attribution-weighted aggregation |
| Whitelist bypass can force source taint to zero | Misconfigured allowlists may mask risk from a whitelisted chunk | Require signed, scoped, expiring, audited whitelist entries and keep injection scanning active |
| Attribution uses n-gram overlap for closed-weight APIs | Lower accuracy on paraphrased adversarial content | Use model-based attribution (self-citation, chain-of-thought) |

---

## 5. Verdict

**Suitable for:** research prototype demonstration, component-level unit testing, policy
engine logic validation, scaffolding for a full evaluation pipeline.

**Not yet suitable for:** venue submission claiming production deployment robustness.
E6 demonstrates an executable end-to-end local RAG pipeline with conditioned
metrics, a 1,000-passage semi-real benign FPR corpus, triaged warnings,
multi-seed confidence intervals, small neural/open-weight checks, and local
baselines. E7 adds FAISS + sentence-transformer retrieval, AG News and Wikitext
real benign corpora, FLAN-T5-small generation for 100 attack queries, a
SmolLM2-135M-Instruct generator-sensitivity run, a three-seed CI run, and one
working external prompt-injection classifier baseline. E8 adds a 150-document
manual red-team set with a 50-case regex-evasive slice, ASR outcome
decomposition, ChromaDB retrieval, Qwen2.5-1.5B-Instruct generation, a 50-case
annotated holdout set, a real local plugin workflow through the tool monitor,
and reproduced LLM Guard plus NeMo Guardrails named baselines. Top-tier
venues still require stronger hosted/GPU-scale open-weight generations, tuned
external guardrail deployments, production SaaS plugin execution, and
independently recruited human-red-team annotation.

**Estimated work to close the gap:** 6–8 weeks of engineering + human annotation.
