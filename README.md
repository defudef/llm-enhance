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
uv run python train_controller.py data/train.jsonl --offline --epochs 1
```

Format `JSONL`:

```json
{"prompt":"Ile to 2+2?","response":"4"}
{"prompt":"Zaplanuj prosty weekend w Krakowie","response":"Sobota: ..."}
```

Inference z wytrenowanym controllerem:

```bash
uv run python infer_controller.py \
  "Ile to 17 * 19?" \
  --controller-checkpoint artifacts/controller.pt \
  --offline
```

## Current Limits

- Ładowanie pełnych wag przy starcie procesu nadal trochę trwa, bo model nie jest trzymany w długowiecznym serwisie.
- Maski attention są nadal referencyjne, nie specjalnie optymalizowane pod bardzo długi kontekst.
- MVP controllera trenuje dane przykład po przykładzie, bez batching/padding pipeline.
