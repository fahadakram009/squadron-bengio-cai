#!/usr/bin/env python3
"""
03_NotebookLM_RAG_Audit.py  --  Mission 5, Part 3 (Data & Payload Specialist)
Google NotebookLM Grounding & Citation Audit Lab

Workflow
--------
1. python 03_NotebookLM_RAG_Audit.py init     -> writes audit_data.json (protocol + empty observations)
2. Run the probes in NotebookLM by hand, fill audit_data.json
   (paste the response, list each citation badge, mark entailed true/false).
3. python 03_NotebookLM_RAG_Audit.py report   -> computes metrics, writes notebooklm_audit_report.md

Metrics
-------
Citation Precision = true entailed citations / total citations generated
Refusal Rate       = correct epistemic refusals / negative probes
Audio score        = mean of Factual Fidelity, Persona Differentiation, Pedagogical Value (1-5)
"""
import json
import sys
from datetime import date
from pathlib import Path

DATA = Path("audit_data.json")
REPORT = Path("notebooklm_audit_report.md")

CORPUS = {
    "A": "Lewis et al. (2020) - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (PDF)",
    "B": "Edge et al. (2024) - From Local to Global: A Graph RAG Approach to Query-Focused Summarization (PDF)",
    "C": "USSE Noosphere Technical Archive Extract (course repo)",
}

# 5 cross-document queries: each must require evidence from >= 2 documents.
# Adapt the Doc C wording to what the archive extract actually contains.
GROUNDED_QUERIES = [
    "Compare how RAG (Doc A) and GraphRAG (Doc B) each address questions that need information from the whole corpus rather than a few passages.",
    "According to Doc A, what is the difference between parametric and non-parametric memory, and how does Doc B's graph index change what the non-parametric memory stores?",
    "How do RAG-Sequence and RAG-Token (Doc A) differ, and which limitation of naive retrieval from Doc B's motivation would each still suffer from?",
    "What role do community summaries play in GraphRAG (Doc B), and how could that be applied to the telemetry records in the USSE archive (Doc C)?",
    "Which failure modes of ungrounded generation do Docs A and B describe, and which of them does the AURA corruption scenario in Doc C resemble?",
]

# 3 plausible but intentionally ABSENT queries. Verify with Ctrl+F in the PDFs/extract before running.
NEGATIVE_QUERIES = [
    "What exact BLEU score did Lewis et al. report for RAG on the WMT14 English-German task?",
    "Which GraphRAG variant in Doc B uses quantum-annealed embeddings, and what speedup did it achieve?",
    "What is the part number of the Hyperion plasma injector coil listed in the USSE archive?",
]

AUDIO_METRICS = ["factual_fidelity", "persona_differentiation", "pedagogical_value"]


def template() -> dict:
    return {
        "meta": {
            "auditor": "",
            "date": str(date.today()),
            "notebook_name": "",
            "sources": CORPUS,
            "notebooklm_version_notes": "",
        },
        "grounded_probes": [
            {
                "id": f"G{i}",
                "query": q,
                "response_text": "",
                "uncited_claims": 0,  # claims in the response with NO citation badge
                "citations": [
                    # {"badge": 1, "claim": "...", "source": "A", "highlighted_span": "...",
                    #  "entailed": true, "note": ""}
                ],
            }
            for i, q in enumerate(GROUNDED_QUERIES, 1)
        ],
        "negative_probes": [
            {
                "id": f"N{i}",
                "query": q,
                "verified_absent_from_sources": False,  # confirm with Ctrl+F first
                "response_text": "",
                "outcome": "",  # "refusal" | "hallucination" | "partial_hedge"
                "note": "",
            }
            for i, q in enumerate(NEGATIVE_QUERIES, 1)
        ],
        "audio_overview": {
            "generated": False,
            "length_minutes": None,
            "excerpt_start_time": "",
            "excerpt_transcript": "",  # ~2 minutes, transcribed
            "unsupported_claims_found": [],
            "scores": {m: None for m in AUDIO_METRICS},
            "justification": {m: "" for m in AUDIO_METRICS},
        },
    }


def citation_precision(probes):
    per_query, total, entailed = [], 0, 0
    for p in probes:
        cites = p["citations"]
        n = len(cites)
        e = sum(1 for c in cites if c.get("entailed") is True)
        per_query.append((p["id"], e, n, e / n if n else None, p.get("uncited_claims", 0)))
        total += n
        entailed += e
    return per_query, entailed, total, (entailed / total if total else None)


def refusal_stats(probes):
    counts = {"refusal": 0, "hallucination": 0, "partial_hedge": 0, "unrecorded": 0}
    for p in probes:
        counts[p["outcome"] if p["outcome"] in counts else "unrecorded"] += 1
    n = len(probes)
    return counts, (counts["refusal"] / n if n else None)


def fmt(x, pct=False):
    if x is None:
        return "PENDING"
    return f"{x:.1%}" if pct else f"{x:.2f}"


def build_report(d: dict) -> str:
    per_q, ent, tot, cp = citation_precision(d["grounded_probes"])
    counts, rr = refusal_stats(d["negative_probes"])
    au = d["audio_overview"]
    scores = [s for s in au["scores"].values() if s is not None]
    audio_mean = sum(scores) / len(scores) if len(scores) == len(AUDIO_METRICS) else None

    L = []
    L.append("# NotebookLM Grounding & Citation Audit Report")
    L.append(f"**Station:** Data & Payload Specialist | **Mission 5, Part 3** | **Date:** {d['meta']['date']}")
    L.append(f"**Auditor:** {d['meta']['auditor'] or 'PENDING'} | **Notebook:** {d['meta']['notebook_name'] or 'PENDING'}\n")

    L.append("## 1. Corpus Ingestion")
    for k, v in CORPUS.items():
        L.append(f"- **Doc {k}:** {v}")
    L.append("")

    L.append("## 2. Grounded Query & Citation Probe")
    L.append("| Query | Entailed | Total | Precision | Uncited claims |")
    L.append("|---|---|---|---|---|")
    for qid, e, n, p, u in per_q:
        L.append(f"| {qid} | {e} | {n} | {fmt(p, True)} | {u} |")
    L.append(f"| **Overall** | **{ent}** | **{tot}** | **{fmt(cp, True)}** | |")
    L.append("\n**Citation Precision = True Entailed Citations / Total Citations Generated**\n")
    for p in d["grounded_probes"]:
        L.append(f"### {p['id']}: {p['query']}")
        L.append(f"**Response:** {p['response_text'] or 'PENDING'}\n")
        if p["citations"]:
            L.append("| Badge | Source | Claim | Highlighted span | Entailed? |")
            L.append("|---|---|---|---|---|")
            for c in p["citations"]:
                span = c.get("highlighted_span", "").replace("|", "\\|").replace("\n", " ")
                claim = c.get("claim", "").replace("|", "\\|")
                L.append(f"| [{c.get('badge')}] | {c.get('source')} | {claim} | {span} | {c.get('entailed')} |")
        L.append("")

    L.append("## 3. Negative Constraint & Epistemic Refusal Probe")
    L.append("| ID | Verified absent | Outcome | Note |")
    L.append("|---|---|---|---|")
    for p in d["negative_probes"]:
        L.append(f"| {p['id']} | {p['verified_absent_from_sources']} | {p['outcome'] or 'PENDING'} | {p['note']} |")
    L.append(f"\n**Correct refusal rate:** {fmt(rr, True)} "
             f"(refusal={counts['refusal']}, hallucination={counts['hallucination']}, "
             f"partial hedge={counts['partial_hedge']}, unrecorded={counts['unrecorded']})\n")
    for p in d["negative_probes"]:
        L.append(f"- **{p['id']}** _{p['query']}_  \n  Response: {p['response_text'] or 'PENDING'}")
    L.append("")

    L.append("## 4. Audio Overview (Deep Dive) Analysis")
    L.append(f"Generated: {au['generated']} | Length: {au['length_minutes'] or 'PENDING'} min | "
             f"Excerpt starts at: {au['excerpt_start_time'] or 'PENDING'}\n")
    L.append("### 2-minute transcript excerpt")
    L.append("> " + (au["excerpt_transcript"].replace("\n", "\n> ") or "PENDING"))
    L.append("\n| Metric (1-5) | Score | Justification |")
    L.append("|---|---|---|")
    for m in AUDIO_METRICS:
        L.append(f"| {m.replace('_', ' ').title()} | {au['scores'][m] if au['scores'][m] is not None else 'PENDING'} "
                 f"| {au['justification'][m] or 'PENDING'} |")
    L.append(f"| **Mean** | **{fmt(audio_mean)}** | |")
    if au["unsupported_claims_found"]:
        L.append("\n**Unsupported claims detected:**")
        L += [f"- {c}" for c in au["unsupported_claims_found"]]
    L.append("")

    L.append("## 5. Conclusions")
    L.append(f"- Citation Precision: {fmt(cp, True)}")
    L.append(f"- Epistemic refusal rate: {fmt(rr, True)}")
    L.append(f"- Audio Overview mean: {fmt(audio_mean)} / 5")
    L.append("- Failure modes observed, limitations (single-vendor black box, chunk-level not claim-level "
             "citations, small n), and implications for AURA Subsystem 05: _write after data collection_.")
    return "\n".join(L) + "\n"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "init":
        if DATA.exists():
            sys.exit(f"{DATA} already exists; refusing to overwrite your observations.")
        DATA.write_text(json.dumps(template(), indent=2, ensure_ascii=False))
        print(f"Wrote {DATA}. Fill it in after running the probes in NotebookLM.")
    elif cmd == "report":
        d = json.loads(DATA.read_text())
        REPORT.write_text(build_report(d))
        print(f"Wrote {REPORT}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
