#!/usr/bin/env python3
"""Build subset/subset-manifest.json with the official stratified sampler.

Calls `src.utils.stratified_indices` exactly as `scripts/experiment.py`
(`load_dataset_split`) does: rows = (language, subject) of the EXAMS-V
validation split, subset_size=200, seed=42, min_per_language=10.

The manifest stores the selected dataset indices, the per-(language, subject)
allocation, and per-question identifiers/metadata/gold labels (no images, no
question text) so every configuration can be checked against it.

    python scripts/make_subset_manifest.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(os.environ["REPO"])
REPL = Path(os.environ["REPL"])
sys.path.insert(0, str(REPO))

from datasets import load_dataset  # noqa: E402

from src.utils import normalize_answer_key, stratified_indices  # noqa: E402

SEED = 42
SIZE = int(os.environ.get("SUBSET_SIZE", 200))
MIN_PER_LANGUAGE = 10


def main() -> None:
    ds = load_dataset(os.environ["DATASET_ID"])["validation"]
    meta = ds.select_columns(["sample_id", "language", "subject", "answer_key", "grade"]) \
        if "grade" in ds.column_names else ds.select_columns(["sample_id", "language", "subject", "answer_key"])
    rows = [{"language": r["language"], "subject": r["subject"]} for r in meta]
    indices, allocation = stratified_indices(rows, subset_size=SIZE, seed=SEED, min_per_language=MIN_PER_LANGUAGE)
    questions = []
    for i in indices:
        r = meta[i]
        questions.append({
            "dataset_index": i,
            "question_id": r["sample_id"],
            "language": r["language"],
            "subject": r["subject"],
            "answer_key_raw": r["answer_key"],
            "gold": normalize_answer_key(r["answer_key"]) if r["answer_key"] else None,
        })
    manifest = {
        "dataset_id": os.environ["DATASET_ID"],
        "dataset_revision": os.environ["DATASET_REVISION"],
        "split": "validation",
        "split_num_rows": ds.num_rows,
        "sampler": "src.utils.subsets.stratified_indices (official repo)",
        "repo_commit": (REPL / "environment" / "repo-commit.txt").read_text().strip(),
        "seed": SEED,
        "subset_size": len(indices),
        "min_per_language": MIN_PER_LANGUAGE,
        "stratify": True,
        "indices": indices,
        "allocation": allocation,
        "language_counts": dict(Counter(q["language"] for q in questions).most_common()),
        "subject_counts": dict(Counter(q["subject"] for q in questions).most_common()),
        "questions": questions,
    }
    out = REPL / "subset" / "subset-manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    # References file in the format expected by the official eval/evaluate.py
    refs = [{"question_id": q["question_id"], "answer": q["gold"], "language": q["language"],
             "subject": q["subject"]} for q in questions]
    (REPL / "subset" / "references.json").write_text(json.dumps(refs, indent=2, ensure_ascii=False))
    print(f"subset: {len(indices)} questions, {len(allocation)} strata, {len(manifest['language_counts'])} languages")
    print(json.dumps(manifest["language_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
