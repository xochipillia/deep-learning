#!/usr/bin/env python3
"""Recompute every reported metric for one result directory from raw records.

    python scripts/compute_metrics.py results/self-consistency-2k [--expected 200]

Reads only files saved inside the package:
  <dir>/raw/**/candidates.jsonl   official per-question records
  <dir>/raw/**/config.yaml        resolved config (max_new_tokens etc.)
  <dir>/raw/subset_*.json         subset written by the official runner
  <dir>/run_meta.json             wall clock + exit code
  <dir>/gpu-memory.csv            nvidia-smi samples
  <dir>/experiment.log            runner log (per-question failures are logged there)
  subset/subset-manifest.json     the seed-42 subset

and writes <dir>/metrics.json, <dir>/predictions.json and
<dir>/official-eval.json (output of the official eval/evaluate.py).

Independent checks performed (results stored under "verification"):
  * gold labels in records == manifest gold labels
  * `correct` flags == (selected_answer == gold)
  * stored extracted answers == official extract_answer(chain text)
  * stored selected answers == official majority_vote(chains)   (SC runs)
  * question ids == the manifest subset (or its first N for smoke tests)
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(os.environ["REPO"])
REPL = Path(os.environ["REPL"])
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.utils import Candidate, extract_answer  # noqa: E402
from src.verify import majority_vote  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import wilson  # noqa: E402

PRICE_H = float(os.environ.get("PRICE_INSTANCE_HOURLY_USD", 2.80))
PRICE_GB_MONTH = float(os.environ.get("PRICE_EBS_GB_MONTH_USD", 0.112))
EBS_GB = float(os.environ.get("EBS_GB", 200))
HOURS_PER_MONTH = 730.0
NO_ANSWER = "NO_ANSWER"   # placeholder so the official evaluator counts it as wrong


def load_official_evaluate():
    spec = importlib.util.spec_from_file_location("official_evaluate", REPO / "eval" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_peak(path: Path) -> dict:
    if not path.exists():
        return {"peak_gpu_memory_used_mib": None, "gpu_memory_total_mib": None, "samples": 0}
    used, total = [], None
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                used.append(float(row[1])); total = float(row[2])
            except ValueError:
                continue
    return {
        "peak_gpu_memory_used_mib": max(used) if used else None,
        "peak_gpu_memory_used_gib": round(max(used) / 1024, 2) if used else None,
        "gpu_memory_total_mib": total,
        "samples": len(used),
        "method": "nvidia-smi memory.used, 500 ms sampling, whole GPU (vLLM pre-allocates "
                  "gpu_memory_utilization=0.9 of memory for weights + KV cache)",
    }


def cost(wall_s: float) -> dict:
    h = wall_s / 3600
    return {
        "instance_hours": round(h, 4),
        "compute_usd": round(h * PRICE_H, 4),
        "ebs_storage_usd_during_run": round(EBS_GB * PRICE_GB_MONTH * h / HOURS_PER_MONTH, 4),
        "price_basis": f"ml.g6e.2xlarge Studio JupyterLab ${PRICE_H}/h; gp3 ${PRICE_GB_MONTH}/GB-month x {EBS_GB:.0f} GB",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--expected", type=int, default=None,
                    help="Expected number of questions (default: manifest size; smoke: 10)")
    args = ap.parse_args()
    d = args.result_dir if args.result_dir.is_absolute() else REPL / args.result_dir
    manifest = json.loads((REPL / "subset" / "subset-manifest.json").read_text())
    mq = {q["question_id"]: q for q in manifest["questions"]}
    expected_n = args.expected or manifest["subset_size"]
    expected_ids = [q["question_id"] for q in manifest["questions"]][:expected_n]

    cand_files = sorted(d.glob("raw/**/candidates.jsonl"))
    if not cand_files:
        raise SystemExit(f"no candidates.jsonl under {d}/raw")
    if len(cand_files) > 1:
        raise SystemExit(f"expected exactly one candidates.jsonl, found {cand_files}")
    records = load_jsonl(cand_files[0])
    cfg_files = sorted(d.glob("raw/**/config.yaml"))
    is_repair = d.name == "guided-repair"
    if is_repair:
        cfg_files = sorted((REPL / "results" / "self-consistency-2k").glob("raw/**/config.yaml"))
    cfg = yaml.safe_load(cfg_files[0].read_text()) if cfg_files else {}
    gen = cfg.get("generation", {})
    max_new = int(gen.get("max_new_tokens", 0)) or None
    run_meta = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
    log = (d / "experiment.log").read_text(errors="replace") if (d / "experiment.log").exists() else ""
    runner_failures = re.findall(r"Failed on (\S+)", log)

    # ---- subset checks ---------------------------------------------------
    ids = [r["question_id"] for r in records]
    subset_files = sorted(d.glob("raw/subset_*.json"))
    runner_subset_match = None
    if subset_files:
        runner_subset_match = all(json.loads(p.read_text())["indices"] == manifest["indices"] for p in subset_files)
    verification = {
        "n_records": len(records),
        "n_expected": expected_n,
        "duplicate_ids": len(ids) - len(set(ids)),
        "ids_equal_expected_subset": sorted(ids) == sorted(expected_ids),
        "missing_ids": sorted(set(expected_ids) - set(ids)),
        "unexpected_ids": sorted(set(ids) - set(expected_ids)),
        "runner_subset_indices_equal_manifest": runner_subset_match,
        "runner_logged_question_failures": runner_failures,
        "runner_exit_code": run_meta.get("exit_code"),
    }

    # ---- per-question recomputation --------------------------------------
    evaluate = load_official_evaluate()
    preds, correct, gold_mismatch, correct_flag_mismatch = [], 0, 0, 0
    extract_mismatch, vote_mismatch, vote_checked = 0, 0, 0
    q_fail, chain_fail, n_chains_total = 0, 0, 0
    trunc_chains, trunc_q, trunc_and_fail = 0, 0, 0
    prompt_unique, prompt_logical, completion, gens = [], [], [], []
    latency = []
    unanimous, disagreement, ties, all8 = 0, 0, 0, 0
    repaired_q, repaired_correct = 0, 0
    for r in records:
        qid = r["question_id"]
        gold = mq.get(qid, {}).get("gold")
        if r.get("gold") != gold:
            gold_mismatch += 1
        sel = r["verifier"].get("selected_answer")
        is_correct = sel is not None and gold is not None and sel == gold
        correct += is_correct
        if bool(r.get("correct")) != is_correct:
            correct_flag_mismatch += 1
        preds.append({"question_id": qid, "predicted_answer": sel or NO_ANSWER})
        if sel is None:
            q_fail += 1
        chains = r.get("chains", [])
        gens.append(len(chains))
        n_chains_total += len(chains)
        if chains:
            prompt_unique.append(int(chains[0].get("prompt_tokens") or 0))
            prompt_logical.append(sum(int(c.get("prompt_tokens") or 0) for c in chains))
            completion.append(sum(int(c.get("completion_tokens") or 0) for c in chains))
        latency.append(float(r.get("total_latency_s") or 0.0))
        any_trunc = False
        for c in chains:
            forced = c.get("phase2_forced", False)
            if not forced and extract_answer(c.get("reasoning", "")) != c.get("extracted_answer"):
                extract_mismatch += 1
            if c.get("extracted_answer") is None:
                chain_fail += 1
            t = max_new is not None and int(c.get("completion_tokens") or 0) >= max_new
            trunc_chains += t
            any_trunc |= t
            if t and (c.get("extracted_answer") is None or forced):
                trunc_and_fail += 1
        trunc_q += any_trunc
        answers = [c.get("extracted_answer") for c in chains]
        valid = [a for a in answers if a]
        if len(chains) > 1:
            if r["verifier"].get("phase2_repair"):
                repaired_q += 1
                repaired_correct += is_correct
                top = Counter(valid).most_common(1)[0][0] if valid else None
                vote_checked += 1
                vote_mismatch += top != sel
            else:
                vote_checked += 1
                cands = [Candidate(description_id=-1, chain_id=c["chain_idx"], description="",
                                   reasoning=c.get("reasoning", ""), answer=c.get("extracted_answer"),
                                   logprob=c.get("logprob_mean")) for c in chains]
                vote_mismatch += majority_vote(cands).answer != sel
            if len(set(valid)) > 1:
                disagreement += 1
            if valid and len(set(valid)) == 1 and len(valid) == len(chains):
                unanimous += 1
            if len(valid) == len(chains):
                all8 += 1
            if r["verifier"].get("tie_break"):
                ties += 1

    official = evaluate(preds, json.loads((REPL / "subset" / "references.json").read_text())[:expected_n]
                        if expected_n != manifest["subset_size"] else
                        json.loads((REPL / "subset" / "references.json").read_text()),
                        stratify_by=["language", "subject"])
    n = len(records)
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    wall = float(run_meta.get("wall_clock_s") or 0.0)

    verification.update({
        "gold_mismatches_vs_manifest": gold_mismatch,
        "correct_flag_mismatches": correct_flag_mismatch,
        "extracted_answer_mismatches_vs_official_extract_answer": extract_mismatch,
        "selected_answer_mismatches_vs_recomputed_vote": vote_mismatch,
        "questions_vote_rechecked": vote_checked,
        "official_evaluate_accuracy_equals_recomputed": abs(official["overall"]["accuracy"] - (correct / n if n else 0)) < 1e-12,
    })
    verification["all_checks_passed"] = (
        verification["duplicate_ids"] == 0 and verification["ids_equal_expected_subset"]
        and verification["runner_subset_indices_equal_manifest"] in (True, None)
        and not runner_failures and run_meta.get("exit_code") in (0, None)
        and gold_mismatch == 0 and correct_flag_mismatch == 0 and extract_mismatch == 0
        and vote_mismatch == 0 and verification["official_evaluate_accuracy_equals_recomputed"]
    )

    metrics = {
        "result": d.name if d.parent.name == "results" else f"{d.parent.name}/{d.name}",
        "config": run_meta.get("config"),
        "model_id": cfg.get("model", {}).get("name", os.environ.get("MODEL_ID")),
        "model_revision": os.environ.get("MODEL_REVISION"),
        "runner": cfg.get("runner"),
        "prompt_style": gen.get("prompt_style"),
        "temperature": gen.get("temperature"),
        "max_new_tokens": max_new,
        "n_chains_config": cfg.get("self_consistency", {}).get("n", 1),
        "accuracy": round(correct / n, 6) if n else None,
        "correct": correct,
        "total": n,
        "accuracy_over_expected": round(correct / expected_n, 6) if expected_n else None,
        "accuracy_ci95_wilson": list(wilson(correct, n)),
        "by_language": {
            lang: {"correct": v["correct"], "total": v["total"], "accuracy": round(v["accuracy"], 6),
                   "ci95_wilson": list(wilson(v["correct"], v["total"]))}
            for lang, v in sorted(official["by_language"].items())
        },
        "answer_extraction": {
            "question_level_failures": q_fail,
            "question_level_failure_rate": round(q_fail / n, 6) if n else None,
            "chain_level_failures": chain_fail,
            "chain_level_failure_rate": round(chain_fail / n_chains_total, 6) if n_chains_total else None,
            "definition": "question-level: no extractable answer in any chain (selected_answer is null); "
                          "chain-level: official extract_answer returned None for that chain",
        },
        "truncation": {
            "chains_at_max_new_tokens": trunc_chains,
            "chain_truncation_rate": round(trunc_chains / n_chains_total, 6) if n_chains_total else None,
            "questions_with_any_truncated_chain": trunc_q,
            "truncated_chains_without_extracted_answer": trunc_and_fail,
            "definition": "completion_tokens >= max_new_tokens (finish_reason is not stored by the official records)",
        },
        "votes": {
            "questions_all_chains_extracted": all8,
            "questions_unanimous": unanimous,
            "questions_with_chain_disagreement": disagreement,
            "questions_with_logprob_tie_break": ties,
        } if n_chains_total > n else None,
        "guided_repair": {"questions_repaired": repaired_q, "repaired_correct": repaired_correct} if is_repair else None,
        "runtime": {
            "wall_clock_s": wall,
            "sum_question_latency_s": round(sum(latency), 2),
            "avg_question_latency_s": mean(latency),
            "avg_wall_clock_per_question_s": round(wall / n, 3) if n else None,
            "overhead_s_model_load_and_dataset": round(wall - sum(latency), 2) if wall else None,
        },
        "tokens": {
            "avg_prompt_tokens_per_question_prefill": mean(prompt_unique),
            "avg_prompt_tokens_per_question_summed_over_chains": mean(prompt_logical),
            "avg_completion_tokens_per_question": mean(completion),
            "avg_completion_tokens_per_chain": round(sum(completion) / n_chains_total, 2) if n_chains_total else None,
            "total_completion_tokens": sum(completion),
            "note": "generate_n reports the same prompt length for each of the N chains; prefill is paid once per question",
        },
        "generations_per_question": mean(gens),
        "gpu": gpu_peak(d / "gpu-memory.csv"),
        "aws_cost_estimate": cost(wall),
        "official_eval_overall": official["overall"],
        "verification": verification,
        "source_records": str(cand_files[0].relative_to(REPL)),
    }
    if is_repair:
        notes = []
        if not records or not repaired_q:
            notes.append("no question required repair")
        sc = json.loads((REPL / "results" / "self-consistency-2k" / "metrics.json").read_text())
        cum = sc["runtime"]["wall_clock_s"] + wall
        metrics["runtime"] = {
            "wall_clock_s": wall,
            "note": "wall_clock_s is the repair pass only (model load + dataset scan + guided decoding of the "
                    "repaired questions); per-question latencies are not recorded by the official repair script",
            "sum_question_latency_s": None,
            "avg_question_latency_s": None,
            "avg_wall_clock_per_question_s": round(wall / n, 3) if n else None,
            "overhead_s_model_load_and_dataset": None,
            "cumulative_wall_clock_s_including_sc2k": round(cum, 2),
            "cumulative_avg_wall_clock_per_question_s": round(cum / n, 3) if n else None,
        }
        metrics["aws_cost_estimate_cumulative_including_sc2k"] = cost(sc["runtime"]["wall_clock_s"] + wall)
        metrics["tokens"]["note"] += ("; token fields are inherited from the SC-2k chains. The official repair script "
                                      "adds one guided token per chain of repaired questions and does not record its prompt tokens")
        metrics["notes"] = notes
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (d / "predictions.json").write_text(json.dumps(preds, indent=2))
    (d / "official-eval.json").write_text(json.dumps(official, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in ("result", "accuracy", "correct", "total")}),
          "checks_passed=", verification["all_checks_passed"])


if __name__ == "__main__":
    main()
