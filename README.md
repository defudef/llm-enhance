# llm-enhance

Minimalny, własny runtime `PyTorch` dla `arcee-ai/Trinity-Nano-Preview` bez używania `transformers` jako silnika modelu. Kod runtime siedzi w pakiecie `trinity/`.

## Setup

```bash
echo 3.13 > .python-version
uv sync
```

## Run

```bash
uv run llm-enhance \
  "Napisz krotkie hello world w Pythonie." \
  --max-new-tokens 16
```

Z controllerem:

```bash
uv run llm-enhance \
  "Ile to 17 * 19?" \
  --with-controller \
  --controller-strength 0.2 \
  --offline
```

Jeśli chcesz podać już sformatowany prompt, użyj `--raw-prompt`.
Domyślnie backend jest wykrywany automatycznie w kolejności `cuda -> mps -> cpu`. `--device` zostaje tylko jako ręczny override.
`--dtype auto` dobiera precision tak:
`cuda -> bfloat16` jeśli wspierane, inaczej `float16`; `mps -> float16`; `cpu -> float32`.
CLI wypisuje postęp na `stderr`, więc przy ciężkim ładowaniu wag i wolniejszej generacji na `mps` nie wygląda jak freeze.
Sama odpowiedź jest streamowana token po tokenie do `stdout`.
Model i tak jest cachowany na dysku przez Hugging Face, domyślnie pod `~/.cache/huggingface/hub/`.
Jeśli chcesz całkiem lokalny workflow bez checków do sieci, użyj:

```bash
uv run llm-enhance "Siema" --offline
```

Jeśli chcesz mieć pliki jawnie zmaterializowane w katalogu projektu:

```bash
uv run llm-enhance "Siema" --local-dir .models/trinity-nano --offline
```

Kompatybilny wrapper dalej działa:

```bash
uv run python main.py "Hello"
```

## Controller MVP

W repo jest też MVP osobnego controllera, który steruje routerem MoE bez modyfikowania checkpointu Trinity.

- controller bierze embedding promptu
- robi pooling promptu
- produkuje biasy `num_moe_layers x num_experts`
- Trinity używa tych biasów przed `topk` w routerze

Trening:

```bash
uv run python train_controller.py --offline --epochs 1
```

Domyślny dataset treningowy to `data/sarcastic_en.jsonl`. Inny dataset możesz nadal podać jako argument pozycyjny, np. `uv run python train_controller.py data/smoke.jsonl --offline`.

Format `JSONL`:

```json
{"prompt":"Ile to 2+2?","response":"4"}
{"prompt":"Zaplanuj prosty weekend w Krakowie","response":"Sobota: ..."}
```

Przykładowy dataset sarkastyczny po angielsku jest w `data/sarcastic_en.jsonl`:

```bash
uv run python train_controller.py \
  --offline \
  --epochs 1 \
  --output-path artifacts/controller-sarcastic-pl-last.pt \
  --best-output-path artifacts/controller-sarcastic-pl-best.pt
```

Przy takim otwartym, stylistycznym dataspecie `val_loss` jest bardziej użyteczny niż exact-match, bo wiele sarkastycznych odpowiedzi może być poprawnych.
`--output-path` zapisuje ostatnią zakończoną epokę, a `--best-output-path` tylko najlepszą metrykę walidacyjną.

Szybsza pętla eksperymentalna:

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

`--dtype float16` przyspiesza bazowy model na MPS, `--max-response-tokens` skraca targety treningowe, a `--eval-every` ogranicza koszt walidacji. Jeśli wrócą NaNy, wróć z `--dtype float16` do domyślnego `float32`.

Inference z wytrenowanym controllerem:

```bash
uv run python infer_controller.py \
  "Ile to 17 * 19?" \
  --controller-strength 0.2 \
  --offline
```

To samo przez główne CLI:

```bash
uv run llm-enhance \
  "Ile to 17 * 19?" \
  --with-controller \
  --controller-strength 0.2 \
  --offline
```

Ewaluacja `base vs controller`:

```bash
uv run python eval_controller.py data/smoke.jsonl --offline
uv run python eval_controller.py data/smoke.jsonl \
  --controller-checkpoint artifacts/controller-best.pt \
  --offline \
  --results-path artifacts/eval.jsonl
```

## IFEval

Do benchmarku instruction-following jest też runner oparty o oficjalny evaluator IFEval z Google Research.

Base model:

```bash
uv run llm-enhance-ifeval \
  --offline \
  --output-dir artifacts/ifeval-base
```

Base vs controller:

```bash
uv run llm-enhance-ifeval \
  --offline \
  --controller-checkpoint artifacts/controller-best.pt \
  --output-dir artifacts/ifeval-controller
```

Przydatne flagi:

- `--max-examples 25` na szybki smoke run
- `--max-new-tokens 512` lub więcej, bo część promptów IFEval wymaga dłuższych odpowiedzi
- `--system-prompt` jeśli chcesz benchmarkować konkretny styl instrukcji systemowej

Wyniki lądują osobno dla `base/` i opcjonalnie `controller/`:

- `responses.jsonl` z wygenerowanymi odpowiedziami
- `eval_results_strict.jsonl`
- `eval_results_loose.jsonl`
- `summary.json` z prompt-level i instruction-level accuracy

## Model Server

Jest też tryb długowiecznego procesu inference, który ładuje bazowy model tylko raz i opcjonalnie cache'uje controllery po ścieżce checkpointu.

Start serwera:

```bash
uv run llm-enhance-serve --offline --port 8000
```

Klient do base model:

```bash
uv run llm-enhance-remote \
  "Ile to 17 * 19?" \
  --server-url http://127.0.0.1:8000
```

Klient z controllerem:

```bash
uv run llm-enhance-remote \
  "Ile to 17 * 19?" \
  --server-url http://127.0.0.1:8000 \
  --controller-checkpoint artifacts/controller-best.pt \
  --controller-strength 0.2
```

Endpointy serwera:

- `GET /healthz`
- `GET /info`
- `POST /generate`

Pełny benchmark z logowaniem na żywo najwygodniej odpalić skryptem:

```bash
./scripts/run_ifeval_full.sh
```

Smoke:

```bash
./scripts/run_ifeval_full.sh --max-examples 10
```

## Gemma 4 E2B

Jest też osobny MVP controllera dla `Gemma 4 E2B`, ale dla dense modelu, więc jako soft-prompt controller zamiast router-bias controller.

Inference z bazową Gemmą:

```bash
uv run llm-enhance-gemma \
  "Write a short joke about saving RAM."
```

Inference z controllerem:

```bash
uv run llm-enhance-gemma \
  "Write a short joke about saving RAM." \
  --controller-checkpoint artifacts/gemma-ifeval-synthetic-best.pt \
  --controller-strength 0.05
```

Dataset pod instruction-following i twardsze formatowanie:

```bash
uv run python scripts/build_gemma_ifeval_synthetic.py
```

Generator zapisuje teraz 576 syntetycznych rekordów IFEval-style. Domyślny
trening Gemmy używa `data/gemma_ifeval_synthetic_en.jsonl`,
zapisuje checkpointy do `artifacts/gemma-ifeval-synthetic*.pt`, loguje MLflow
i ma ustawione hparamy pod bieżący eksperyment IFEval. Wystarczy:

```bash
uv run python gemma_train_controller.py
```

Aktualne defaulty treningu: `--epochs 24`, `--learning-rate 2e-4`,
`--warmup-ratio 0.03`, `--num-virtual-tokens 16`, `--controller-dim 256`,
`--controller-hidden-dim 1024`, `--controller-dropout 0`,
`--max-response-tokens 512`, `--val-split 0.2`, `--patience 10` i `--mlflow`.

MLflow zapisuje lokalną bazę runów domyślnie w `artifacts/mlflow.db`,
a artefakty w `artifacts/mlflow-artifacts`. UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

Opis architektury MVP jest w:

- `docs/gemma-e2b-controller.md`

IFEval dla surowej Gemmy:

```bash
uv run llm-enhance-gemma-ifeval \
  --base-only \
  --output-dir artifacts/gemma-ifeval
```

W trakcie runu CLI loguje `elapsed`, średni czas na prompt i `eta`, a do katalogu wyniku zapisuje na bieżąco:

- `responses.jsonl`
- `progress.json`
- `summary.partial.json`
- `eval_results_strict.partial.jsonl`
- `eval_results_loose.partial.jsonl`

Domyślnie partial snapshot odświeża się co `10` promptów. Możesz to zmienić przez `--save-every`.

IFEval `base vs controller` dla Gemmy:

```bash
uv run llm-enhance-gemma-ifeval
```

## Current Limits

- Ładowanie pełnych wag przy starcie procesu nadal trochę trwa, bo model nie jest trzymany w długowiecznym serwisie.
- Maski attention są nadal referencyjne, nie specjalnie optymalizowane pod bardzo długi kontekst.
- MVP controllera trenuje dane przykład po przykładzie, bez batching/padding pipeline.
