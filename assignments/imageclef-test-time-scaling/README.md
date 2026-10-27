# ImageCLEF Visual MCQ Test-Time Scaling

Controlled reproduction of the method described in *More Tokens, Fewer
Trees: Test-Time Scaling for Small VLMs on Multilingual Visual MCQ*.

The implementation uses the official
[`lesterpjy/tts-small-vlm`](https://github.com/lesterpjy/tts-small-vlm)
repository with a frozen Qwen3.5-4B model. It does not train or publish a new
checkpoint.

## Compared Configurations

| Configuration | Accuracy | 95% Wilson interval |
| --- | ---: | ---: |
| Zero-shot | 59.0% | 52.1-65.6% |
| Self-consistency, `N=8`, 1,024 tokens | 80.0% | 73.9-85.0% |
| Self-consistency, `N=8`, 2,048 tokens | 79.0% | 72.8-84.1% |
| 2,048 tokens with guided repair | 79.5% | 73.4-84.5% |

These values come from one stratified 200-question validation subset with
seed 42. The intervals overlap, so the small difference between the 1,024-
and 2,048-token runs is descriptive rather than conclusive.

The full aggregate tables are in
[`results/aggregate-results.csv`](results/aggregate-results.csv) and
[`results/by-language-aggregate.csv`](results/by-language-aggregate.csv).

## Repository Contents

```text
run-all.ipynb       end-to-end notebook with cleared outputs
configs/            zero-shot and self-consistency experiment definitions
scripts/            setup, execution, evaluation, and hygiene utilities
results/            aggregate metrics only
```

## Reproduction

Requirements:

- Linux with an NVIDIA GPU with at least 40 GB VRAM
- Python 3.12
- Git
- sufficient disk space for the model and EXAMS-V cache

The notebook is the primary entry point:

1. Open `run-all.ipynb`.
2. Review the parameters cell.
3. Run all cells in order.

The setup script clones the official repository at commit
`2247c130075889fcdcac31e0eb83dba40c67afbd` and downloads pinned model and
dataset revisions. Generated artifacts are written to ignored directories.

The scripts use the following public resources:

- Model: `Qwen/Qwen3.5-4B`
- Dataset: `MBZUAI/EXAMS-V`, validation split
- Paper: [arXiv:2607.09438](https://arxiv.org/abs/2607.09438)

Review the model and dataset licenses and terms before downloading or using
them.

## Publication Boundaries

This repository intentionally excludes:

- dataset files and images;
- raw question text, options, identifiers, and answer keys;
- per-question predictions and generated reasoning;
- executed notebooks, logs, environment dumps, and cloud resource records;
- model weights and caches;
- credentials, cloud account identifiers, local paths, and personal data;
- the teammate-authored comparison notebook;
- presentation and course-submission files.

The included aggregate metrics cannot be used to reconstruct individual
dataset records.

## Attribution

The experiment wrappers call the official implementation rather than
reimplementing the model pipeline. Refer to the original repository and paper
for the method, prompts, and evaluation definitions.
