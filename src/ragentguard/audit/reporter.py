"""
Stage 6: Audit Report Generator.

Produces structured audit reports from accumulated PolicyViolations.
Output formats:
  - JSONL log (one record per violation, appended in real-time)
  - JSON summary (aggregated per-session report)
  - HTML report (human-readable, suitable for paper figures)

Report schema (per violation):
  - source documents that contributed to the violation
  - full retrieval-to-action taint chain
  - attack category classification
  - suggested mitigations
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.provenance import AttackCategory, PolicyViolation


_MITIGATION_MAP: Dict[AttackCategory, List[str]] = {
    AttackCategory.RETRIEVAL_INJECTION: [
        "Remove or quarantine flagged source documents from the corpus.",
        "Enable StruQ-style structural separation in the prompt builder.",
        "Increase context_untrust_threshold to reduce false negatives.",
    ],
    AttackCategory.MEMORY_POISONING: [
        "Audit long-term memory entries originating from low-trust sources.",
        "Set a trust-level TTL: expire untrusted documents after N sessions.",
        "Re-embed corpus after removing flagged documents.",
    ],
    AttackCategory.JUDGE_MANIPULATION: [
        "Use a separate, isolated retrieval corpus for the judge component.",
        "Cross-validate judge scores against a reference judge without retrieval.",
        "Quarantine documents with judge_manipulation patterns.",
    ],
    AttackCategory.CROSS_TOOL_TAINT: [
        "Enforce strict tool input sanitization for all external tool calls.",
        "Log and review multi-hop tool chains where taint score > 0.3.",
        "Require human approval for tool calls with taint_score > 0.5.",
    ],
    AttackCategory.UNKNOWN: [
        "Investigate flagged documents manually.",
        "Lower policy thresholds for stricter detection.",
    ],
}


def _mitigations_for(violation: PolicyViolation) -> List[str]:
    cat = violation.taint_vector.detected_attack_category
    return _MITIGATION_MAP.get(cat, _MITIGATION_MAP[AttackCategory.UNKNOWN])


class AuditLogger:
    """
    Real-time JSONL logger: appends one JSON record per violation.
    Suitable for streaming audit logs in production deployments.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log(self, violation: PolicyViolation) -> None:
        record = {
            **violation.to_dict(),
            "mitigations": _mitigations_for(violation),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class AuditReporter:
    """
    Session-level audit reporter.

    Accumulates violations across a full pipeline session, then writes:
      1. A JSON summary report
      2. A human-readable HTML report
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._violations: List[PolicyViolation] = []
        self._session_start = datetime.utcnow()
        self._config = config_dict or {}

    def record(self, violation: PolicyViolation) -> None:
        self._violations.append(violation)

    def record_many(self, violations: List[PolicyViolation]) -> None:
        self._violations.extend(violations)

    # ------------------------------------------------------------------ #
    # JSON summary                                                         #
    # ------------------------------------------------------------------ #

    def to_json(self) -> Dict[str, Any]:
        """Aggregate report as a Python dict (JSON-serialisable)."""
        by_category: Dict[str, int] = {}
        blocked_count = 0
        affected_sources: Dict[str, int] = {}

        for v in self._violations:
            cat = v.taint_vector.detected_attack_category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            if v.blocked:
                blocked_count += 1
            for src in v.taint_vector.unique_source_ids:
                affected_sources[src] = affected_sources.get(src, 0) + 1

        return {
            "session_start": self._session_start.isoformat(),
            "session_end": datetime.utcnow().isoformat(),
            "total_violations": len(self._violations),
            "blocked_count": blocked_count,
            "violations_by_category": by_category,
            "top_offending_sources": sorted(
                affected_sources.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "violations": [
                {
                    **v.to_dict(),
                    "mitigations": _mitigations_for(v),
                }
                for v in self._violations
            ],
        }

    def write_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    # ------------------------------------------------------------------ #
    # HTML report                                                          #
    # ------------------------------------------------------------------ #

    def write_html(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        summary = self.to_json()
        html = self._render_html(summary)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def _render_html(self, summary: Dict[str, Any]) -> str:
        cat_rows = "".join(
            f"<tr><td>{cat}</td><td>{count}</td></tr>"
            for cat, count in summary["violations_by_category"].items()
        )
        src_rows = "".join(
            f"<tr><td><code>{src}</code></td><td>{cnt}</td></tr>"
            for src, cnt in summary["top_offending_sources"]
        )

        violation_cards = ""
        for v in summary["violations"]:
            mitigations_li = "".join(f"<li>{m}</li>" for m in v.get("mitigations", []))
            sources_li = "".join(
                f"<li>{s.get('source_path','?')} (trust={s.get('trust_level','?'):.2f})</li>"
                for s in v.get("taint_vector", {}).get("contributing_sources", [])
            )
            blocked_badge = (
                '<span class="badge blocked">BLOCKED</span>'
                if v.get("blocked")
                else '<span class="badge flagged">FLAGGED</span>'
            )
            violation_cards += f"""
            <div class="violation-card">
              <h3>{blocked_badge} [{v.get('stage','?')}] {v.get('violation_type','?')}</h3>
              <p><strong>Timestamp:</strong> {v.get('timestamp','?')}</p>
              <p><strong>Attack category:</strong>
                 {v.get('taint_vector',{}).get('attack_category','?')}</p>
              <p><strong>Query:</strong> <em>{v.get('query','')[:200]}</em></p>
              <p><strong>Context fraction from untrusted sources:</strong>
                 {v.get('context_fraction',0):.1%}</p>
              <p><strong>Taint score:</strong>
                 {v.get('taint_vector',{}).get('taint_score',0):.3f}</p>
              <details>
                <summary>Taint chain</summary>
                <pre>{' → '.join(v.get('taint_vector',{}).get('propagation_path',[]))}</pre>
              </details>
              <details>
                <summary>Contributing sources ({len(v.get('taint_vector',{}).get('contributing_sources',[]))})</summary>
                <ul>{sources_li}</ul>
              </details>
              <details>
                <summary>Injection patterns detected</summary>
                <ul>{''.join(f'<li>{p}</li>' for p in v.get('taint_vector',{}).get('injection_patterns',[]))}</ul>
              </details>
              <details>
                <summary>Suggested mitigations</summary>
                <ul>{mitigations_li}</ul>
              </details>
            </div>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAGentGuard Audit Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: auto; padding: 2rem; }}
  h1 {{ color: #c0392b; }}
  h2 {{ border-bottom: 2px solid #eee; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  .violation-card {{ border: 1px solid #ddd; border-radius: 6px;
                     padding: 1rem; margin-bottom: 1rem; background: #fafafa; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.8rem; font-weight: bold; }}
  .badge.blocked {{ background: #e74c3c; color: white; }}
  .badge.flagged  {{ background: #f39c12; color: white; }}
  summary {{ cursor: pointer; font-weight: bold; }}
  pre {{ background: #f0f0f0; padding: 0.5rem; border-radius: 4px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>RAGentGuard Audit Report</h1>
<p>Session: {summary['session_start']} → {summary['session_end']}</p>

<h2>Summary</h2>
<table>
  <tr><th>Total violations</th><td>{summary['total_violations']}</td></tr>
  <tr><th>Blocked</th><td>{summary['blocked_count']}</td></tr>
  <tr><th>Flagged (review)</th><td>{summary['total_violations'] - summary['blocked_count']}</td></tr>
</table>

<h2>Violations by Attack Category</h2>
<table>
  <tr><th>Category</th><th>Count</th></tr>
  {cat_rows}
</table>

<h2>Top Offending Sources</h2>
<table>
  <tr><th>Source ID</th><th>Violation count</th></tr>
  {src_rows}
</table>

<h2>Violation Details</h2>
{violation_cards if violation_cards else '<p>No violations recorded.</p>'}
</body>
</html>"""

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def reset(self) -> None:
        self._violations.clear()
        self._session_start = datetime.utcnow()
