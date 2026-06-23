"""
Run the default E1-E6 evaluation experiments and print summary tables.
Set RUN_E7_REAL_STACK=1 to run the heavier optional E7 neural retrieval
experiment after the default artifact run.
Set RUN_E8_REVIEWER_EXTRAS=1 to run the manual red-team and sandboxed
tool-execution reviewer extras.

Metric columns:
  ASR        : Attack Success Rate (lower = better)
  Warn-R     : warning_recall  — any violation fired (warn OR block)
  Block-R    : blocking_recall — operationally blocked (≤ Warn-R)
  Inj-R      : injection_recall — structural pattern detected
  bFPR       : blocked_fpr    — benign queries blocked (operational FP)
  wFPR       : warning_fpr    — benign queries warned (informational)
  Attr%      : attribution_accuracy (valid in E3 only; "N/A" elsewhere)
  Lat(ms)    : mean latency per query

Run:
    python examples/run_experiments.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragentguard.evaluation import (
    run_e1_single_vs_crossstage,
    run_e2_category_coverage,
    run_e3_attribution_accuracy,
    run_e4_latency,
    run_e5_realworld_simulation,
    run_e6_end_to_end_rag,
    run_e6_multiseed,
    run_e7_real_stack,
    run_e7_multiseed,
    run_e8_human_redteam_rag,
    run_e8_stronger_generator_rag,
    run_e8_chromadb_rag,
    run_e8_independent_annotated_redteam_rag,
    run_e8_adaptive_redteam_rag,
    run_e8_regex_evasive_generator_stress,
    run_e8_semantic_threshold_calibration,
    run_e8_semantic_roc_analysis,
    run_e8_sandboxed_tool_execution,
    run_e8_real_plugin_tool_workflow,
    run_e9_hybrid_calibrated_defense,
    run_e9_full_neural_hybrid_defense,
    run_e9_tuned_baseline_parity,
    run_e9_aggregation_ablation,
    run_e9_scaled_operator_cost_study,
)


def print_table(title: str, results: dict) -> None:
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    hdr = (
        f"{'System/Category':<28} {'ASR':>6} {'Warn-R':>8} {'Block-R':>9}"
        f" {'Inj-R':>7} {'bFPR':>6} {'wFPR':>6} {'Attr%':>7} {'Lat(ms)':>9}"
    )
    print(hdr)
    print("-" * 90)
    for name, res in results.items():
        r = res.to_dict()
        attr = r["attribution_accuracy"]
        attr_str = f"{attr:>7.2%}" if isinstance(attr, float) else f"{'N/A':>7}"
        print(
            f"{name:<28} "
            f"{r['attack_success_rate']:>6.2%} "
            f"{r['warning_recall']:>8.2%} "
            f"{r['blocking_recall']:>9.2%} "
            f"{r['injection_recall']:>7.2%} "
            f"{r['blocked_fpr']:>6.2%} "
            f"{r['warning_fpr']:>6.2%} "
            f"{attr_str} "
            f"{r['mean_latency_ms']:>9.2f}"
        )


def main():
    os.makedirs("./output/experiments", exist_ok=True)

    print("\nRunning E1: Single-stage vs. cross-stage...")
    e1 = run_e1_single_vs_crossstage(n_per_category=50)
    print_table("E1: Single-stage vs. Cross-stage Defense", e1)
    print("  NOTE: Attr% = N/A in E1; attribution measured only in E3.")
    with open("./output/experiments/e1_results.json", "w") as f:
        json.dump({k: v.to_dict() for k, v in e1.items()}, f, indent=2)

    print("\nRunning E2: Attack category coverage...")
    e2 = run_e2_category_coverage(n_per_category=80)
    print_table("E2: RAGentGuard Recall by Attack Category", e2)
    print("  NOTE: memory_poisoning has injection_recall=0 by design (no structural patterns).")
    print("  NOTE: blocking_recall = injection_recall for all categories (pattern = block).")
    with open("./output/experiments/e2_results.json", "w") as f:
        json.dump({k: v.to_dict() for k, v in e2.items()}, f, indent=2)

    print("\nRunning E3: Attribution accuracy (multi-source context)...")
    e3 = run_e3_attribution_accuracy(n_queries=80)
    print_table("E3: Attribution Accuracy (1 adversarial + 4 distractors)", e3)
    print("  NOTE: output reuses adversarial doc text — see SOURCE_VALIDATION_REPORT §2.4.")
    with open("./output/experiments/e3_results.json", "w") as f:
        json.dump({k: v.to_dict() for k, v in e3.items()}, f, indent=2)

    print("\nRunning E4: Latency overhead...")
    e4 = run_e4_latency(n_queries=100)
    r4 = e4.to_dict()
    print(f"\n{'='*90}")
    print("  E4: Latency Overhead (Python policy engine only, no I/O)")
    print(f"{'='*90}")
    print(f"  Mean latency: {r4['mean_latency_ms']:.3f} ms   P95: {r4['p95_latency_ms']:.3f} ms")
    print("  NOTE: excludes vector DB retrieval, LLM inference, and audit log I/O.")
    with open("./output/experiments/e4_results.json", "w") as f:
        json.dump(r4, f, indent=2)

    print("\nRunning E5: Real-world app simulation...")
    e5 = run_e5_realworld_simulation()
    print_table("E5: Real-World RAG App Simulation", e5)
    print("  NOTE: wFPR shows trust_risk warnings on LOW-trust benign docs (expected).")
    with open("./output/experiments/e5_results.json", "w") as f:
        json.dump({k: v.to_dict() for k, v in e5.items()}, f, indent=2)

    print("\nRunning E6: End-to-end local RAG pipeline...")
    e6 = run_e6_end_to_end_rag(n_per_category=8, n_benign=1000, n_benign_queries=200, top_k=5)
    r6 = e6.to_dict()
    print(f"\n{'='*90}")
    print("  E6: End-to-End Local RAG Evaluation")
    print(f"{'='*90}")
    print(f"  Attack queries       : {r6['n_attack_queries']}")
    print(f"  Benign queries       : {r6['n_benign_queries']}")
    print(f"  Retrieval hit rate   : {r6['retrieval_hit_rate']:.2%}")
    print(f"  Context warn recall  : {r6['context_warning_recall']:.2%}")
    print(f"  Context block recall : {r6['context_blocking_recall']:.2%}")
    print(f"  Cond. warn recall    : {r6['conditioned_context_warning_recall']:.2%}")
    print(f"  Cond. block recall   : {r6['conditioned_context_blocking_recall']:.2%}")
    print(f"  Cond. attack success : {r6['conditioned_attack_success_rate']:.2%}")
    print(f"  Attack success rate  : {r6['attack_success_rate']:.2%}")
    print(f"  Actionable warn rec. : {r6['actionable_warning_recall']:.2%}")
    print(f"  Tool attempt rate    : {r6['tool_attempt_rate']:.2%}")
    print(f"  Tool block rate      : {r6['tool_block_rate']:.2%}")
    print(f"  Shadow tool attempts : {r6['shadow_tool_attempt_rate']:.2%}")
    print(f"  Shadow tool blocks   : {r6['shadow_tool_block_rate']:.2%}")
    print(f"  Benign blocked FPR   : {r6['benign_blocked_fpr']:.2%}")
    print(f"  Benign warning FPR   : {r6['benign_warning_fpr']:.2%}")
    print(f"  Benign action FPR    : {r6['benign_actionable_warning_fpr']:.2%}")
    print(f"  Mean / P95 latency   : {r6['mean_latency_ms']:.2f} / {r6['p95_latency_ms']:.2f} ms")
    print(f"  Embedding / generator: {r6['embedding_backend']} / {r6['generator_backend']}")
    print(f"  Benign corpus        : {r6['benign_corpus']}")
    print("  Baselines            :")
    for baseline, metrics in r6["baselines"].items():
        print(
            f"    {baseline:<17} block={metrics['blocking_recall']:.2%} "
            f"ASR={metrics['attack_success_rate']:.2%} "
            f"bFPR={metrics['benign_blocked_fpr']:.2%}"
        )
    print("  Category metrics     :")
    for category, metrics in r6["category_metrics"].items():
        print(
            f"    {category:<20} hit={metrics['retrieval_hit_rate']:.2%} "
            f"block={metrics['context_blocking_recall']:.2%} "
            f"cond_block={metrics['conditioned_blocking_recall']:.2%} "
            f"ASR={metrics['attack_success_rate']:.2%}"
        )
    print("  NOTE: E6 uses a SQLite-backed local vector store and deterministic local generator by default.")
    print("  NOTE: Shadow tool metrics ask what the tool gate would do if blocked contexts continued.")
    with open("./output/experiments/e6_results.json", "w") as f:
        json.dump(r6, f, indent=2)
    with open("./output/experiments/e6_traces.json", "w") as f:
        json.dump([trace.__dict__ for trace in e6.traces[:20]], f, indent=2)

    print("\nRunning E6: Multi-seed confidence intervals...")
    e6_multi = run_e6_multiseed(
        seeds=(40, 41, 42, 43, 44),
        n_per_category=8,
        n_benign=1000,
        n_benign_queries=200,
        top_k=5,
    )
    r6m = e6_multi.to_dict()
    print(f"\n{'='*90}")
    print("  E6: Multi-Seed Summary (mean ± 95% CI)")
    print(f"{'='*90}")
    for name, metric in r6m["metrics"].items():
        print(f"  {name:<38}: {metric['mean']:.4f} ± {metric['ci95']:.4f}")
    with open("./output/experiments/e6_multiseed_results.json", "w") as f:
        json.dump(r6m, f, indent=2)

    if os.environ.get("RUN_E7_REAL_STACK") == "1":
        print("\nRunning E7: Real-stack FAISS/sentence-transformer RAG pipeline...")
        e7 = run_e7_real_stack()
        r7 = e7.to_dict()
        print(f"\n{'='*90}")
        print("  E7: Real-Stack RAG Evaluation")
        print(f"{'='*90}")
        print(f"  Attack / benign queries : {r7['n_attack_queries']} / {r7['n_benign_queries']}")
        print(f"  Retrieval hit rate      : {r7['retrieval_hit_rate']:.2%}")
        print(f"  Context block recall    : {r7['context_blocking_recall']:.2%}")
        print(f"  Cond. block recall      : {r7['conditioned_context_blocking_recall']:.2%}")
        print(f"  Attack success rate     : {r7['attack_success_rate']:.2%}")
        print(f"  Cond. attack success    : {r7['conditioned_attack_success_rate']:.2%}")
        print(f"  Benign blocked FPR      : {r7['benign_blocked_fpr']:.2%}")
        print(f"  Benign action FPR       : {r7['benign_actionable_warning_fpr']:.2%}")
        print(f"  Mean / P95 latency      : {r7['mean_latency_ms']:.2f} / {r7['p95_latency_ms']:.2f} ms")
        print(f"  Embedding / generator   : {r7['embedding_backend']} / {r7['generator_backend']}")
        print("  Baselines               :")
        for baseline, metrics in r7["baselines"].items():
            print(
                f"    {baseline:<17} block={metrics['blocking_recall']:.2%} "
                f"ASR={metrics['attack_success_rate']:.2%} "
                f"bFPR={metrics['benign_blocked_fpr']:.2%}"
            )
        with open("./output/experiments/e7_real_stack_results.json", "w") as f:
            json.dump(r7, f, indent=2)

    if os.environ.get("RUN_E7_FLAN_CI") == "1":
        guard_model = os.environ.get(
            "E7_EXTERNAL_GUARD_MODEL",
            "testsavantai/prompt-injection-defender-tiny-v0",
        )
        generator_backend = os.environ.get("E7_GENERATOR_BACKEND", "transformers")
        generator_model = os.environ.get("E7_GENERATOR_MODEL", "google/flan-t5-small")
        print("\nRunning E7: open-weight generation with CI and external guard baseline...")
        e7_flan = run_e7_real_stack(
            n_mutated_attacks=68,
            n_benign=5000,
            n_benign_queries=500,
            generator_backend=generator_backend,
            generator_model=generator_model,
            benign_dataset=os.environ.get("E7_BENIGN_DATASET", "ag_news"),
            external_guard_model=guard_model,
        )
        r7f = e7_flan.to_dict()
        with open("./output/experiments/e7_flan_100_results.json", "w") as f:
            json.dump(r7f, f, indent=2)
        e7_ci = run_e7_multiseed(
            seeds=(40, 41, 42),
            n_mutated_attacks=68,
            n_benign=5000,
            n_benign_queries=500,
            generator_backend=generator_backend,
            generator_model=generator_model,
            benign_dataset=os.environ.get("E7_BENIGN_DATASET", "ag_news"),
            external_guard_model=guard_model,
        )
        r7ci = e7_ci.to_dict()
        with open("./output/experiments/e7_flan_100_multiseed_results.json", "w") as f:
            json.dump(r7ci, f, indent=2)

    if os.environ.get("RUN_E8_REVIEWER_EXTRAS") == "1":
        print("\nRunning E8a: human-red-team RAG evaluation...")
        e8_red = run_e8_human_redteam_rag()
        with open("./output/experiments/e8_human_redteam_results.json", "w") as f:
            json.dump(e8_red.to_dict(), f, indent=2)
        if os.environ.get("RUN_E8_REGEX_STRESS") == "1":
            print("\nRunning E8c: regex-evasive generator stress evaluation...")
            e8_regex = run_e8_regex_evasive_generator_stress()
            with open("./output/experiments/e8_regex_evasive_smolm2_results.json", "w") as f:
                json.dump(e8_regex.to_dict(), f, indent=2)
        if os.environ.get("RUN_E8_STRONG_GENERATOR") == "1":
            print("\nRunning E8e: stronger generator evaluation...")
            e8_strong = run_e8_stronger_generator_rag(
                generator_model=os.environ.get(
                    "E8_STRONG_GENERATOR_MODEL",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                )
            )
            with open("./output/experiments/e8_qwen25_15b_results.json", "w") as f:
                json.dump(e8_strong.to_dict(), f, indent=2)
        if os.environ.get("RUN_E8_CHROMA") == "1":
            print("\nRunning E8f: ChromaDB backend evaluation...")
            e8_chroma = run_e8_chromadb_rag()
            with open("./output/experiments/e8_chromadb_results.json", "w") as f:
                json.dump(e8_chroma.to_dict(), f, indent=2)
        if os.environ.get("RUN_E8_INDEPENDENT_REDTEAM") == "1":
            print("\nRunning E8g: independent annotated red-team holdout...")
            e8_ind = run_e8_independent_annotated_redteam_rag()
            with open("./output/experiments/e8_independent_annotated_redteam_results.json", "w") as f:
                json.dump(e8_ind.to_dict(), f, indent=2)
        if os.environ.get("RUN_E8_ADAPTIVE_REDTEAM") == "1":
            print("\nRunning E8i: adaptive policy-aware red-team evaluation...")
            e8_adaptive = run_e8_adaptive_redteam_rag()
            with open("./output/experiments/e8_adaptive_redteam_results.json", "w") as f:
                json.dump(e8_adaptive.to_dict(), f, indent=2)
        if os.environ.get("RUN_E8_SEMANTIC_CALIBRATION") == "1":
            print("\nRunning E8d: semantic threshold calibration...")
            e8_semantic = run_e8_semantic_threshold_calibration()
            with open("./output/experiments/e8_semantic_threshold_calibration.json", "w") as f:
                json.dump(e8_semantic, f, indent=2)
            print("\nRunning E8j: semantic ROC/domain calibration...")
            e8_semantic_roc = run_e8_semantic_roc_analysis()
            with open("./output/experiments/e8_semantic_roc_analysis.json", "w") as f:
                json.dump(e8_semantic_roc, f, indent=2)
        print("\nRunning E8b: sandboxed real-tool execution evaluation...")
        e8_tools = run_e8_sandboxed_tool_execution()
        with open("./output/experiments/e8_sandboxed_tool_results.json", "w") as f:
            json.dump(e8_tools.to_dict(), f, indent=2)
        if os.environ.get("RUN_E8_REAL_PLUGIN_TOOLS") == "1":
            print("\nRunning E8h: real local plugin/tool workflow evaluation...")
            e8_plugins = run_e8_real_plugin_tool_workflow()
            with open("./output/experiments/e8_real_plugin_tool_workflow_results.json", "w") as f:
                json.dump(e8_plugins.to_dict(), f, indent=2)

    if os.environ.get("RUN_E9_REVIEWER_EXTRAS") == "1":
        print("\nRunning E9a: calibrated hybrid structural+semantic defense...")
        e9_hybrid = run_e9_hybrid_calibrated_defense()
        with open("./output/experiments/e9_hybrid_calibrated_defense.json", "w") as f:
            json.dump(e9_hybrid, f, indent=2)

        if os.environ.get("RUN_E9_FULL_NEURAL") == "1":
            print("\nRunning E9e: full neural FAISS/sentence-transformer hybrid defense...")
            e9_full_neural = run_e9_full_neural_hybrid_defense()
            with open("./output/experiments/e9_full_neural_hybrid_defense.json", "w") as f:
                json.dump(e9_full_neural, f, indent=2)

        print("\nRunning E9b: tuned same-trace baseline parity...")
        e9_parity = run_e9_tuned_baseline_parity()
        with open("./output/experiments/e9_tuned_baseline_parity.json", "w") as f:
            json.dump(e9_parity, f, indent=2)

        print("\nRunning E9c: aggregation ablation...")
        e9_aggregation = run_e9_aggregation_ablation()
        with open("./output/experiments/e9_aggregation_ablation.json", "w") as f:
            json.dump(e9_aggregation, f, indent=2)

        print("\nRunning E9d: scaled operator-cost study...")
        e9_operator = run_e9_scaled_operator_cost_study()
        with open("./output/experiments/e9_scaled_operator_cost.json", "w") as f:
            json.dump(e9_operator, f, indent=2)

    print(f"\n\nAll results saved to ./output/experiments/")


if __name__ == "__main__":
    main()
