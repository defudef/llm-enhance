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

## Current Limits

- Ładowanie pełnych wag przy starcie procesu nadal trochę trwa, bo model nie jest trzymany w długowiecznym serwisie.
- Maski attention są nadal referencyjne, nie specjalnie optymalizowane pod bardzo długi kontekst.
- MVP controllera trenuje dane przykład po przykładzie, bez batching/padding pipeline.
