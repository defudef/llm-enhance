# llm-enhance

Experimental controllers and evaluation harnesses for local LLM inference.

The current focus is a lightweight soft-prompt controller for
`google/gemma-4-E2B-it`, evaluated with IFEval. The repository also contains an
older custom PyTorch runtime for `arcee-ai/Trinity-Nano-Preview` and a
router-bias controller for that MoE model.

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
uses a soft-prompt controller instead of the Trinity router-bias controller.

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

## Trinity Runtime

The repository still includes a minimal custom PyTorch runtime for
`arcee-ai/Trinity-Nano-Preview`, without using `transformers` as the model
execution engine. The runtime code lives under `trinity/`.

Run Trinity:

```bash
uv run llm-enhance \
  "Napisz krotkie hello world w Pythonie." \
  --max-new-tokens 16
```

Run Trinity with the router-bias controller:

```bash
uv run llm-enhance \
  "Ile to 17 * 19?" \
  --with-controller \
  --controller-strength 0.2 \
  --offline
```

Use `--raw-prompt` to pass an already formatted prompt. By default, the backend
is selected in this order: `cuda -> mps -> cpu`. `--dtype auto` selects
`bfloat16` on CUDA when supported, otherwise `float16`; `float16` on MPS; and
`float32` on CPU. The generated answer streams token by token to `stdout`, while
status messages go to `stderr`.

Use offline mode once the model is cached locally:

```bash
uv run llm-enhance "Siema" --offline
```

Use a project-local model directory:

```bash
uv run llm-enhance "Siema" --local-dir .models/trinity-nano --offline
```

The compatibility wrapper still works:

```bash
uv run python main.py "Hello"
```

## Trinity Controller

The Trinity controller is a small router-bias controller for the MoE runtime:

- it reads prompt embeddings,
- pools the prompt,
- produces `num_moe_layers x num_experts` router biases,
- and applies those biases before `topk` routing.

Train it on the default `data/sarcastic_en.jsonl` dataset:

```bash
uv run python train_controller.py --offline --epochs 1
```

Use a custom JSONL dataset:

```bash
uv run python train_controller.py data/smoke.jsonl --offline
```

Dataset format:

```json
{"prompt":"Ile to 2+2?","response":"4"}
{"prompt":"Zaplanuj prosty weekend w Krakowie","response":"Sobota: ..."}
```

Example longer training run:

```bash
uv run python train_controller.py \
  --offline \
  --epochs 1 \
  --output-path artifacts/controller-sarcastic-pl-last.pt \
  --best-output-path artifacts/controller-sarcastic-pl-best.pt
```

Faster experimental loop:

```bash
uv run python train_controller.py \
  --offline \
  --epochs 10 \
  --dtype float16 \
  --controller-bias-scale 0.2 \
  --learning-rate 1e-5 \
  --router-bias-l2 0 \
  --max-response-tokens 48 \
  --eval-every 3
```

`--dtype float16` speeds up the base model on MPS, `--max-response-tokens`
shortens training targets, and `--eval-every` reduces validation cost. If NaNs
return, switch back from `--dtype float16` to the default `float32`.

Run inference with a trained Trinity controller:

```bash
uv run python infer_controller.py \
  "Ile to 17 * 19?" \
  --controller-strength 0.2 \
  --offline
```

Evaluate base vs controller:

```bash
uv run python eval_controller.py data/smoke.jsonl --offline
uv run python eval_controller.py data/smoke.jsonl \
  --controller-checkpoint artifacts/controller-best.pt \
  --offline \
  --results-path artifacts/eval.jsonl
```

## Trinity IFEval

Run the older Trinity IFEval runner on the base model:

```bash
uv run llm-enhance-ifeval \
  --offline \
  --output-dir artifacts/ifeval-base
```

Run base vs controller:

```bash
uv run llm-enhance-ifeval \
  --offline \
  --controller-checkpoint artifacts/controller-best.pt \
  --output-dir artifacts/ifeval-controller
```

Useful flags:

- `--max-examples 25` for a smoke run
- `--max-new-tokens 512` or more for longer IFEval answers
- `--system-prompt` to benchmark a specific system instruction

The helper script runs the full Trinity IFEval with live logging:

```bash
./scripts/run_ifeval_full.sh
```

Smoke run:

```bash
./scripts/run_ifeval_full.sh --max-examples 10
```

## Model Server

The server mode keeps the base model loaded and can cache controller checkpoints
by path.

Start the server:

```bash
uv run llm-enhance-serve --offline --port 8000
```

Call the base model:

```bash
uv run llm-enhance-remote \
  "Ile to 17 * 19?" \
  --server-url http://127.0.0.1:8000
```

Call the model with a controller:

```bash
uv run llm-enhance-remote \
  "Ile to 17 * 19?" \
  --server-url http://127.0.0.1:8000 \
  --controller-checkpoint artifacts/controller-best.pt \
  --controller-strength 0.2
```

Endpoints:

- `GET /healthz`
- `GET /info`
- `POST /generate`

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
