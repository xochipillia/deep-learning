#!/usr/bin/env python3
"""Export reviewer examples (examples/examples.json + examples/examples.md).

Selection is deterministic (first matching question in subset order) and is
done purely from saved records:
  correct            SC-2k question answered correctly
  incorrect          SC-2k question answered incorrectly
  majority_vote_all8 SC-2k question where all 8 chains produced a letter
  chain_disagreement SC-2k question whose chains disagree (prefers non-unanimous majority)
  extraction_failure chain-level (and question-level if any) extraction failures, SC-2k then SC-1k
  repaired           question changed by the official guided repair (if any)
  truncation_1k_fixed_2k  question with truncated, unextracted chains at 1,024 tokens
                     and no truncated chain at 2,048 tokens (prefers wrong->right)

    python scripts/export_examples.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

REPL = Path(os.environ["REPL"])
RES = REPL / "results"


def load(name: str) -> dict[str, dict]:
    files = sorted((RES / name).glob("raw/**/candidates.jsonl"))
    if not files:
        return {}
    return {r["question_id"]: r for r in map(json.loads, files[0].read_text().splitlines()) if r}


def max_tokens(name: str) -> int | None:
    p = RES / name / "metrics.json"
    return json.loads(p.read_text()).get("max_new_tokens") if p.exists() else None


def view(rec: dict, source: str, max_new: int | None) -> dict:
    chains = rec.get("chains", [])
    return {
        "source_result": source,
        "question_id": rec["question_id"],
        "language": rec.get("language"),
        "subject": rec.get("subject"),
        "gold": rec.get("gold"),
        "prediction": rec["verifier"].get("selected_answer"),
        "correct": rec.get("correct"),
        "extracted_answers": [c.get("extracted_answer") for c in chains],
        "vote_counts": rec["verifier"].get("cluster_sizes"),
        "confidence": rec["verifier"].get("confidence"),
        "tie_break": rec["verifier"].get("tie_break"),
        "phase2_repair": rec["verifier"].get("phase2_repair", False),
        "chains": [{
            "chain_idx": c.get("chain_idx"),
            "extracted_answer": c.get("extracted_answer"),
            "completion_tokens": c.get("completion_tokens"),
            "prompt_tokens": c.get("prompt_tokens"),
            "truncated_at_max_new_tokens": bool(max_new and (c.get("completion_tokens") or 0) >= max_new),
            "phase2_forced": c.get("phase2_forced", False),
            "response": c.get("reasoning"),
        } for c in chains],
    }


def main() -> None:
    order = [q["question_id"] for q in json.loads((REPL / "subset" / "subset-manifest.json").read_text())["questions"]]
    zs, sc1, sc2, rep = load("zero-shot"), load("self-consistency-1k"), load("self-consistency-2k"), load("guided-repair")
    m1, m2 = max_tokens("self-consistency-1k"), max_tokens("self-consistency-2k")
    ex: dict[str, list[dict]] = {}

    def first(recs: dict, pred, k=1):
        return [recs[q] for q in order if q in recs and pred(recs[q])][:k]

    valid = lambda r: [c["extracted_answer"] for c in r["chains"] if c.get("extracted_answer")]  # noqa: E731
    ex["correct"] = [view(r, "self-consistency-2k", m2) for r in first(sc2, lambda r: r.get("correct"))]
    ex["incorrect"] = [view(r, "self-consistency-2k", m2) for r in first(sc2, lambda r: not r.get("correct"))]
    ex["zero_shot_correct_and_incorrect"] = (
        [view(r, "zero-shot", None) for r in first(zs, lambda r: r.get("correct"))]
        + [view(r, "zero-shot", None) for r in first(zs, lambda r: not r.get("correct"))])
    ex["majority_vote_all8"] = [view(r, "self-consistency-2k", m2) for r in first(
        sc2, lambda r: len(valid(r)) == len(r["chains"]) == 8 and len(set(valid(r))) > 1)
        or first(sc2, lambda r: len(valid(r)) == len(r["chains"]) == 8)]
    dis = first(sc2, lambda r: len(set(valid(r))) > 1 and Counter(valid(r)).most_common(1)[0][1] <= len(valid(r)) // 2 + 1)
    ex["chain_disagreement"] = [view(r, "self-consistency-2k", m2) for r in (dis or first(sc2, lambda r: len(set(valid(r))) > 1))]
    fails = []
    for name, recs, mx in (("self-consistency-2k", sc2, m2), ("self-consistency-1k", sc1, m1)):
        qf = first(recs, lambda r: r["verifier"].get("selected_answer") is None)
        cf = first(recs, lambda r: r["verifier"].get("selected_answer") is not None
                   and any(c.get("extracted_answer") is None for c in r["chains"]))
        fails += [view(r, name, mx) | {"failure_level": "question"} for r in qf]
        fails += [view(r, name, mx) | {"failure_level": "chain"} for r in cf]
    ex["extraction_failure"] = fails
    ex["repaired"] = [view(r, "guided-repair", m2) | {"before_repair": view(sc2[r["question_id"]], "self-consistency-2k", m2)}
                      for r in first(rep, lambda r: r["verifier"].get("phase2_repair"), k=3)]
    trunc = []
    if m1 and m2:
        def t1(r):
            return [c for c in r["chains"] if (c.get("completion_tokens") or 0) >= m1 and c.get("extracted_answer") is None]
        cand = [q for q in order if q in sc1 and q in sc2 and t1(sc1[q])
                and not any((c.get("completion_tokens") or 0) >= m2 for c in sc2[q]["chains"])]
        cand.sort(key=lambda q: (not (not sc1[q].get("correct") and sc2[q].get("correct")), -len(t1(sc1[q])), order.index(q)))
        for q in cand[:2]:
            trunc.append({"question_id": q, "at_1k": view(sc1[q], "self-consistency-1k", m1),
                          "at_2k": view(sc2[q], "self-consistency-2k", m2),
                          "truncated_unextracted_chains_at_1k": len(t1(sc1[q])),
                          "n_candidates_meeting_criterion": len(cand)})
    ex["truncation_1k_fixed_2k"] = trunc
    out = REPL / "examples"
    out.mkdir(exist_ok=True)
    (out / "examples.json").write_text(json.dumps(ex, indent=2, ensure_ascii=False))

    def md_rec(v: dict, depth="###") -> list[str]:
        lines = [f"{depth} `{v['question_id']}` ({v['source_result']})", "",
                 f"- Language: {v['language']}; subject: {v['subject']}",
                 f"- Gold: {v['gold']}; prediction: {v['prediction']}; correct: {v['correct']}",
                 f"- Extracted answers: {v['extracted_answers']}",
                 f"- Vote counts: {v['vote_counts']}; confidence: {v['confidence']}; tie-break: {v['tie_break']}"]
        if v.get("phase2_repair"):
            lines.append("- Repaired by official guided repair: yes")
        lines.append("")
        for c in v["chains"]:
            resp = (c["response"] or "")
            tail = resp if len(resp) <= 700 else "…" + resp[-700:]
            lines += [f"<details><summary>chain {c['chain_idx']}: answer={c['extracted_answer']}, "
                      f"completion_tokens={c['completion_tokens']}, truncated={c['truncated_at_max_new_tokens']}"
                      f"{', forced' if c['phase2_forced'] else ''}</summary>", "", "```text", tail, "```", "", "</details>", ""]
        return lines

    md = ["# Examples for review", "",
          "Generated by `scripts/export_examples.py` from the saved records. Full responses are in `examples.json`;",
          "this file shows at most the last 700 characters of each response. Images are not included",
          "(look them up in EXAMS-V validation by `question_id`).", ""]
    titles = {"correct": "Correct (SC-2k)", "incorrect": "Incorrect (SC-2k)",
              "zero_shot_correct_and_incorrect": "Zero-shot correct and incorrect",
              "majority_vote_all8": "Majority vote with all eight extracted answers (SC-2k)",
              "chain_disagreement": "Disagreement between chains (SC-2k)",
              "extraction_failure": "Answer-extraction failures", "repaired": "Guided repair",
              "truncation_1k_fixed_2k": "Truncation at 1,024 tokens that disappears at 2,048"}
    for key, title in titles.items():
        md += [f"## {title}", ""]
        if not ex[key]:
            md += ["None found in the saved records.", ""]
        for v in ex[key]:
            if key == "truncation_1k_fixed_2k":
                md += [f"### `{v['question_id']}`", "",
                       f"{v['truncated_unextracted_chains_at_1k']} chains hit 1,024 tokens without an answer; no chain hit "
                       f"2,048 tokens. Questions meeting this criterion: {v['n_candidates_meeting_criterion']}.", ""]
                md += md_rec(v["at_1k"], "####") + md_rec(v["at_2k"], "####")
            elif key == "repaired":
                md += md_rec(v["before_repair"]) + md_rec(v)
            else:
                md += ([f"Failure level: {v['failure_level']}", ""] if "failure_level" in v else []) + md_rec(v)
    (out / "examples.md").write_text("\n".join(md) + "\n")
    print({k: len(v) for k, v in ex.items()})


if __name__ == "__main__":
    main()
