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

## Current Limits

- Generacja na razie działa bez KV cache, więc każdy kolejny token przelicza cały kontekst od nowa.
- Maski attention są referencyjne, nie zoptymalizowane pod długi kontekst.
- Domyślne `dtype` na `mps` to `float16`, bo jest bardziej przewidywalne niż `bfloat16`.
