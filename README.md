# llm-enhance

Experimental controllers and evaluation harnesses for local LLM inference.

The current focus is a lightweight soft-prompt controller for
`google/gemma-4-E2B-it`, evaluated with IFEval.

Status: experimental. The code is intended to make controller experiments,
generated responses, and IFEval comparisons reproducible. It should not be read
as a validated training method or a performance-optimized inference stack.

## What This Tests

- Whether a small learned controller can steer a frozen local model without
  updating the base weights.
- Whether repeating the same IFEval prompt at generation time changes
  instruction-following accuracy.
- Which instruction categories improve or regress under controller and prompt
  ablations.

The benchmark artifacts are written under `artifacts/` and include raw
responses, strict/loose IFEval outputs, partial progress snapshots, and summary
JSON files.

## Setup

```bash
echo 3.13 > .python-version
uv sync
```

## Gemma 4 E2B

This is the main active experiment. `Gemma 4 E2B` is a dense model, so this path
uses a soft-prompt controller.

Run base Gemma:

```bash
uv run llm-enhance-gemma \
  "Write a short joke about saving RAM."
```

Run Gemma with a controller checkpoint:

```bash
uv run llm-enhance-gemma \
  "Write a short joke about saving RAM." \
  --controller-checkpoint artifacts/gemma-ifeval-synthetic-best.pt \
  --controller-strength 0.05
```

Build the synthetic instruction-following dataset:

```bash
uv run python scripts/build_gemma_ifeval_synthetic.py
```

The generator writes `data/gemma_ifeval_synthetic_en.jsonl`. The default Gemma
controller training flow uses that dataset, writes checkpoints to
`artifacts/gemma-ifeval-alignfix-small*.pt`, and logs to MLflow:

```bash
uv run python gemma_train_controller.py
```

Current training defaults:

- `--epochs 8`
- `--learning-rate 5e-5`
- `--grad-accum-steps 8`
- `--warmup-ratio 0.03`
- `--num-virtual-tokens 4`
- `--controller-dim 64`
- `--controller-hidden-dim 256`
- `--controller-dropout 0.05`
- `--max-response-tokens 512`
- `--val-split 0.2`
- `--patience 2`
- `--mlflow`

MLflow stores the local run database at `artifacts/mlflow.db` and artifacts under
`artifacts/mlflow-artifacts` by default:

```bash
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

The controller design note is in `docs/gemma-e2b-controller.md`.

## Gemma IFEval

Run raw Gemma on the full IFEval dataset:

```bash
uv run llm-enhance-gemma-ifeval \
  --base-only \
  --output-dir artifacts/gemma-ifeval
```

Run base vs controller:

```bash
uv run llm-enhance-gemma-ifeval \
  --output-dir artifacts/gemma-ifeval-controller-1x
```

Run only the controller variant when a matching base run already exists:

```bash
uv run llm-enhance-gemma-ifeval \
  --controller-only \
  --controller-checkpoint artifacts/gemma-ifeval-alignfix-small-best.pt \
  --controller-strength 0.05 \
  --output-dir artifacts/gemma-ifeval-controller-1x
```

Run raw Gemma with each prompt repeated twice at generation time:

```bash
uv run llm-enhance-gemma-ifeval \
  --base-only \
  --prompt-repeats 2 \
  --output-dir artifacts/gemma-ifeval-base-2x
```

Run base vs controller with each prompt repeated twice:

```bash
uv run llm-enhance-gemma-ifeval \
  --prompt-repeats 2 \
  --output-dir artifacts/gemma-ifeval-controller-2x
```

`--prompt-repeats 2` changes only the generation prompt. IFEval scoring still
uses the original prompt from the official input set as the key, so the metrics
stay comparable.

During a run, the CLI logs elapsed time, average time per prompt, and ETA. Each
output directory receives:

- `responses.jsonl`
- `progress.json`
- `run_config.json`
- `summary.partial.json`
- `eval_results_strict.partial.jsonl`
- `eval_results_loose.partial.jsonl`
- `summary.json` after the full run completes
- `eval_results_strict.jsonl` after the full run completes
- `eval_results_loose.jsonl` after the full run completes

Partial snapshots refresh every 10 prompts by default. Change that with
`--save-every`.

Interpretation rule: compare full runs against full runs, and partial runs only
against the same prefix of the official IFEval input set. The dataset order is
not guaranteed to have uniform difficulty across prefixes.

Full-run results from the current experiment:

| Variant | Final | Delta vs base 1x | Strict prompt | Strict instruction | Loose prompt | Loose instruction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base 1x | 0.7326 | +0.00pp | 0.6765 | 0.7614 | 0.7061 | 0.7866 |
| Controller 1x | 0.7346 | +0.20pp | 0.6765 | 0.7650 | 0.7079 | 0.7890 |
| Base 2x | 0.7579 | +2.52pp | 0.6987 | 0.7854 | 0.7357 | 0.8118 |
| Controller 2x | 0.7616 | +2.89pp | 0.7043 | 0.7818 | 0.7449 | 0.8153 |

In this run, prompt repetition produced most of the lift: `base 2x` improved
`+2.52pp` over `base 1x`, while `controller 2x` improved only another `+0.37pp`
over `base 2x`. The controller alone was nearly flat at `+0.20pp`.

## Current Limits

- The Gemma runtime is correctness-oriented, not optimized for throughput. On
  MPS it uses a custom token-by-token loop and is much slower than a dedicated
  inference backend such as MLX, llama.cpp, vLLM, or a vendor serving stack.
- Loading full model weights still takes time because the basic CLI paths do not
  keep a long-lived model service warm.
- Attention masks and controller integration are reference implementations, not
  optimized paths for very long context.
- The controller training loop still processes examples one by one, without a
  batching/padding pipeline.

## License And Attribution

This repository is licensed under Apache-2.0.

`google/gemma-4-E2B-it` is published by Google DeepMind under Apache-2.0. If you
redistribute model-derived artifacts, preserve the relevant license and
attribution notices and do not present the work as endorsed by Google.
