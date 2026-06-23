#!/usr/bin/env python3
"""Evaluate NeMo Guardrails injection detection on exported RAG traces."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from nemoguardrails import RailsConfig
from nemoguardrails.library.injection_detection.actions import injection_detection


DEFAULT_INJECTIONS = ("code", "sqli", "template", "xss")


def _rails_config(injections: tuple[str, ...]) -> RailsConfig:
    return RailsConfig.from_content(
        config={
            "rails": {
                "config": {
                    "injection_detection": {
                        "injections": list(injections),
                        "action": "reject",
                    }
                }
            }
        }
    )


async def _scan_contexts(
    traces: list[dict[str, Any]],
    config: RailsConfig,
    max_chars: int,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    start = time.perf_counter()
    for trace in traces:
        context = (trace.get("assembled_context") or "")[:max_chars]
        result = await injection_detection(text=context, config=config)
        blocked = bool(result["is_injection"])
        rows.append(
            {
                "query": trace.get("query", ""),
                "target_category": trace.get("target_category", ""),
                "attack_style": trace.get("attack_style", ""),
                "blocked": blocked,
                "detections": list(result.get("detections", [])),
            }
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return rows, elapsed_ms


def evaluate_trace_file(
    trace_path: Path,
    output_path: Path,
    injections: tuple[str, ...] = DEFAULT_INJECTIONS,
    max_chars: int = 4000,
) -> dict[str, Any]:
    traces = json.loads(trace_path.read_text())
    config = _rails_config(injections)
    rows, elapsed_ms = asyncio.run(_scan_contexts(traces, config, max_chars))

    attack_traces = [trace for trace in traces if trace.get("target_category") != "benign"]
    benign_traces = [trace for trace in traces if trace.get("target_category") == "benign"]
    attack_rows = [row for row in rows if row["target_category"] != "benign"]
    benign_rows = [row for row in rows if row["target_category"] == "benign"]

    blocked_attacks = sum(int(row["blocked"]) for row in attack_rows)
    blocked_benign = sum(int(row["blocked"]) for row in benign_rows)
    attack_successes = sum(
        int((not row["blocked"]) and bool(trace.get("attack_like_generation")))
        for trace, row in zip(traces, rows)
        if trace.get("target_category") != "benign"
    )

    style_counts: dict[str, dict[str, int]] = {}
    for trace, row in zip(traces, rows):
        if trace.get("target_category") == "benign":
            continue
        style = trace.get("attack_style") or "unknown"
        style_counts.setdefault(style, {"n": 0, "blocked": 0, "succeeded": 0})
        style_counts[style]["n"] += 1
        style_counts[style]["blocked"] += int(row["blocked"])
        style_counts[style]["succeeded"] += int(
            (not row["blocked"]) and bool(trace.get("attack_like_generation"))
        )

    detection_counts = Counter(
        detection
        for row in rows
        for detection in row.get("detections", [])
    )
    result = {
        "experiment": "nemo-guardrails-injection-detection-baseline",
        "scanner": "nemoguardrails.library.injection_detection.actions.injection_detection",
        "version": importlib.metadata.version("nemoguardrails"),
        "injections": list(injections),
        "trace_file": str(trace_path),
        "max_chars": max_chars,
        "n_attack_queries": len(attack_traces),
        "n_benign_queries": len(benign_traces),
        "blocking_recall": round(blocked_attacks / len(attack_traces), 4) if attack_traces else 0.0,
        "attack_success_rate": round(attack_successes / len(attack_traces), 4) if attack_traces else 0.0,
        "benign_blocked_fpr": round(blocked_benign / len(benign_traces), 4) if benign_traces else 0.0,
        "mean_scan_latency_ms": round(elapsed_ms / max(1, len(rows)), 2),
        "detection_counts": dict(sorted(detection_counts.items())),
        "attack_style_metrics": {
            style: {
                "n": counts["n"],
                "blocking_recall": round(counts["blocked"] / counts["n"], 4)
                if counts["n"] else 0.0,
                "attack_success_rate": round(counts["succeeded"] / counts["n"], 4)
                if counts["n"] else 0.0,
            }
            for style, counts in sorted(style_counts.items())
        },
        "rows": rows,
    }
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--injections", nargs="+", default=list(DEFAULT_INJECTIONS))
    args = parser.parse_args()
    result = evaluate_trace_file(
        args.trace_path,
        args.output_path,
        injections=tuple(args.injections),
        max_chars=args.max_chars,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
