from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "gemma_prompt_repeat_diverse_en.jsonl"
)
SEED = 29
PROMPTS_PER_CATEGORY = 40


def take_balanced(category: str, prompts: list[str]) -> list[dict[str, str]]:
    unique_prompts = sorted(set(prompt.strip() for prompt in prompts if prompt.strip()))
    if len(unique_prompts) < PROMPTS_PER_CATEGORY:
        raise ValueError(
            f"Category {category} produced {len(unique_prompts)} prompts; "
            f"need at least {PROMPTS_PER_CATEGORY}."
        )
    rng = random.Random(f"{SEED}:{category}")
    rng.shuffle(unique_prompts)
    return [
        {
            "prompt": prompt,
            "category": category,
            "source": "prompt_repeat_diverse_v1",
        }
        for prompt in unique_prompts[:PROMPTS_PER_CATEGORY]
    ]


def build_reasoning_prompts() -> list[dict[str, str]]:
    scenarios = [
        "a team can ship either a small fix today or a larger refactor next week",
        "a researcher sees an accuracy gain but the sample is only 80 examples",
        "a warehouse has two packing lines and one line starts failing intermittently",
        "a mobile app gets faster but battery use increases",
        "a teacher has to choose between a quiz and a longer project",
        "an API cache reduces latency but sometimes serves stale records",
        "a startup can optimize onboarding or improve reliability this sprint",
        "a model benchmark improves after the prompt is repeated",
    ]
    questions = [
        "list the most likely tradeoffs",
        "identify the strongest counterargument",
        "give a decision with two caveats",
        "explain what evidence would change the decision",
        "rank three next actions",
    ]
    return take_balanced(
        "reasoning",
        [f"For this situation, {question}: {scenario}." for scenario in scenarios for question in questions],
    )


def build_math_prompts() -> list[dict[str, str]]:
    items = [
        ("notebooks", 7, 13, 4),
        ("GPU hours", 12, 9, 5),
        ("support tickets", 18, 6, 11),
        ("training batches", 5, 24, 17),
        ("warehouse boxes", 9, 15, 8),
        ("daily commits", 11, 7, 19),
        ("dataset shards", 4, 28, 6),
        ("evaluation runs", 6, 14, 10),
    ]
    templates = [
        "Solve the word problem step by step: We have {a} {item}, add {b}, then remove {c}. What remains?",
        "Compute the final count. Start with {a} {item}, multiply by {b}, then subtract {c}.",
        "A log shows {a} groups of {b} {item} and {c} extra. What is the total?",
        "Estimate and then calculate exactly: {a} teams each use {b} {item}; {c} are unused.",
        "Give the arithmetic expression and answer for {a}, {b}, and {c} in the {item} scenario.",
    ]
    return take_balanced(
        "math_word_problem",
        [
            template.format(item=item, a=a, b=b, c=c)
            for item, a, b, c in items
            for template in templates
        ],
    )


def build_code_prompts() -> list[dict[str, str]]:
    languages = ["Python", "TypeScript", "Rust", "SQL", "Bash", "Go", "JavaScript", "Swift"]
    tasks = [
        "parse a JSONL file and count rows by category",
        "retry a flaky HTTP request with exponential backoff",
        "deduplicate records while preserving order",
        "validate an email address without external services",
        "stream a large file without loading it all into memory",
        "compute rolling averages for a metric series",
        "normalize user-provided file paths safely",
        "format a compact progress message for a CLI",
    ]
    prompts = []
    for language in languages:
        for task in tasks:
            prompts.append(f"Write a concise {language} function to {task}. Include one edge case.")
            prompts.append(f"Review this planned {language} implementation for {task}; list likely bugs first.")
    return take_balanced("code", prompts)


def build_data_prompts() -> list[dict[str, str]]:
    payloads = [
        "name=Ana, city=Krakow, score=91",
        "model=gemma, split=validation, final=0.7616",
        "host=api-2, status=degraded, latency_ms=420",
        "order=1842, currency=PLN, total=129.50",
        "sensor=temp, value=22.4, unit=C",
        "task=cleanup, branch=work-next, owner=research",
        "file=summary.json, rows=541, passed=true",
        "user=leo, plan=pro, renewal=2026-05-01",
    ]
    formats = ["JSON", "CSV", "a Markdown table", "YAML", "key bullet points"]
    return take_balanced(
        "data_transform",
        [
            f"Convert this record into {fmt}: {payload}."
            for payload in payloads
            for fmt in formats
        ],
    )


def build_writing_prompts() -> list[dict[str, str]]:
    audiences = ["researchers", "product managers", "backend engineers", "new users", "investors"]
    topics = [
        "a benchmark result that improved after prompt repetition",
        "a small open-source release",
        "a slow inference pipeline",
        "a cleanup that removed legacy code",
        "a dataset quality concern",
        "a model evaluation caveat",
        "an experiment that needs replication",
        "a tradeoff between speed and accuracy",
    ]
    tones = ["direct", "skeptical", "friendly", "formal", "plainspoken"]
    return take_balanced(
        "writing",
        [
            f"Write a {tone} paragraph for {audience} about {topic}."
            for audience in audiences
            for topic in topics
            for tone in tones
        ],
    )


def build_summarization_prompts() -> list[dict[str, str]]:
    snippets = [
        "The first run finished with better loose instruction accuracy, but strict prompt accuracy barely moved.",
        "The deployment reduced median latency while p95 stayed high during cache misses.",
        "The user interview showed confusion around setup, naming, and where benchmark artifacts are written.",
        "The report says the model is useful for drafting but needs careful factual verification.",
        "The team removed a service because the local CLI path made the server redundant.",
        "The experiment generated many artifacts, but only the final summaries should be compared.",
        "The bug appears when paths are relative and the process starts outside the repository root.",
        "The new dataset is intentionally broad and should not mirror any single benchmark distribution.",
    ]
    styles = ["one sentence", "three bullets", "a risk summary", "a decision note", "a title and subtitle"]
    return take_balanced(
        "summarization",
        [f"Summarize this as {style}: {snippet}" for snippet in snippets for style in styles],
    )


def build_translation_prompts() -> list[dict[str, str]]:
    sentences = [
        "The benchmark is still running, but the partial result is useful.",
        "Please keep the answer short and avoid marketing language.",
        "The model loads correctly, but generation is slower than expected.",
        "We should compare the controller only after the full run finishes.",
        "This branch removes old code and keeps the Gemma path.",
        "The dataset should be broad rather than optimized for one benchmark.",
        "The prompt was repeated twice during generation.",
        "The training objective should be easy to inspect.",
    ]
    targets = ["Polish", "Spanish", "German", "French", "plain English", "technical English"]
    return take_balanced(
        "translation_rewrite",
        [f"Translate or rewrite into {target}: {sentence}" for sentence in sentences for target in targets],
    )


def build_planning_prompts() -> list[dict[str, str]]:
    goals = [
        "run a model evaluation overnight",
        "prepare a small open-source PR",
        "debug a slow local inference loop",
        "design a diverse prompt dataset",
        "migrate a package namespace",
        "compare two benchmark variants",
        "audit a README for stale instructions",
        "ship a minimal experiment without overbuilding",
    ]
    horizons = ["one hour", "one day", "one week", "two sprints", "a weekend"]
    return take_balanced(
        "planning",
        [f"Make a practical {horizon} plan to {goal}." for goal in goals for horizon in horizons],
    )


def build_classification_prompts() -> list[dict[str, str]]:
    texts = [
        "The run improved accuracy, but the sample is too small.",
        "Can you update the README before opening the PR?",
        "The API returned 500 three times after deploy.",
        "I like the idea, but the current loss may be too weak.",
        "The user asked for a concise Polish explanation.",
        "The training data accidentally copied the benchmark format.",
        "The checkout flow is confusing on mobile.",
        "The model output ignored the requested JSON schema.",
    ]
    label_sets = [
        "bug, task, question, observation",
        "positive, negative, mixed, neutral",
        "research, engineering, product, support",
        "urgent, normal, low-priority",
        "risk, decision, action, context",
    ]
    return take_balanced(
        "classification",
        [f"Classify the text into one of these labels: {labels}. Text: {text}" for text in texts for labels in label_sets],
    )


def build_extraction_prompts() -> list[dict[str, str]]:
    notes = [
        "Meeting: 2026-04-12, owner: Marta, decision: rerun base 2x.",
        "Incident 418: service api, severity high, started 09:42 UTC.",
        "Experiment: prompt-repeat, model Gemma, metric final_score, value 0.7616.",
        "Invoice A-17: vendor Northwind, amount 320 USD, due Friday.",
        "Task: update docs, assignee Leo, branch chore-cleanup.",
        "Run log: examples 541, device mps, dtype float16, status complete.",
        "Customer: Nova Labs, plan enterprise, renewal 2026-06-30.",
        "Dataset: diverse prompts, categories 12, split validation 20 percent.",
    ]
    fields = ["date and owner", "ids and status", "metrics", "amounts", "assignee and branch"]
    return take_balanced(
        "extraction",
        [f"Extract {field} from this note and return compact JSON: {note}" for note in notes for field in fields],
    )


def build_creative_prompts() -> list[dict[str, str]]:
    subjects = [
        "a model that learns from echoes",
        "a lighthouse for lost packets",
        "a calendar that negotiates deadlines",
        "a compiler that writes postcards",
        "a robot that audits README files",
        "a garden grown from log lines",
        "a library where functions sleep",
        "a train powered by benchmark results",
    ]
    forms = ["micro-story", "dialogue", "poem", "scene outline", "game item description"]
    return take_balanced(
        "creative",
        [f"Write a concise {form} about {subject}." for subject in subjects for form in forms],
    )


def build_domain_prompts() -> list[dict[str, str]]:
    domains = ["medicine", "law", "finance", "education", "climate science", "security", "operations", "design"]
    tasks = [
        "explain the main uncertainty",
        "list what an expert should verify",
        "rewrite for a non-specialist",
        "separate facts from assumptions",
        "identify a safe next step",
    ]
    return take_balanced(
        "domain_explanation",
        [f"In the context of {domain}, {task} for a short research note." for domain in domains for task in tasks],
    )


def build_records() -> list[dict[str, str]]:
    records = [
        *build_reasoning_prompts(),
        *build_math_prompts(),
        *build_code_prompts(),
        *build_data_prompts(),
        *build_writing_prompts(),
        *build_summarization_prompts(),
        *build_translation_prompts(),
        *build_planning_prompts(),
        *build_classification_prompts(),
        *build_extraction_prompts(),
        *build_creative_prompts(),
        *build_domain_prompts(),
    ]
    random.Random(SEED).shuffle(records)
    validate_records(records)
    return records


def validate_records(records: list[dict[str, str]]) -> None:
    if not records:
        raise ValueError("Dataset is empty.")
    prompts = [record["prompt"] for record in records]
    if len(prompts) != len(set(prompts)):
        raise ValueError("Dataset contains duplicate prompts.")
    invalid_response_records = [record for record in records if "response" in record]
    if invalid_response_records:
        raise ValueError("Prompt-repeat dataset must not contain response targets.")
    counts = Counter(record["category"] for record in records)
    if set(counts.values()) != {PROMPTS_PER_CATEGORY}:
        raise ValueError(f"Unbalanced categories: {dict(sorted(counts.items()))}")


def main() -> None:
    records = build_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
