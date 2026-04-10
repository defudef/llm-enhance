from __future__ import annotations

import json
import random
from pathlib import Path


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "gemma_format_smoke_en.jsonl"
)
SEED = 17


def add_record(
    records: list[dict[str, str]],
    *,
    prompt: str,
    response: str,
    category: str,
) -> None:
    records.append(
        {
            "prompt": prompt,
            "response": response,
            "category": category,
        }
    )


def build_digit_only_records() -> list[dict[str, str]]:
    templates = [
        "Answer with digits only. What is {expr}?",
        "Reply using digits only and nothing else. Compute {expr}.",
        "Give only the numeric answer. Solve {expr}.",
    ]
    examples = [
        ("12 + 7", "19"),
        ("14 + 15", "29"),
        ("33 + 9", "42"),
        ("48 + 16", "64"),
        ("90 - 27", "63"),
        ("81 - 19", "62"),
        ("144 / 12", "12"),
        ("13 * 6", "78"),
        ("11 * 11", "121"),
        ("7 * 9", "63"),
        ("19 + 24", "43"),
        ("125 - 48", "77"),
        ("18 * 4", "72"),
        ("84 / 7", "12"),
        ("96 - 18", "78"),
        ("17 + 25", "42"),
        ("45 + 27", "72"),
        ("63 - 21", "42"),
        ("24 * 3", "72"),
        ("108 / 9", "12"),
        ("56 + 8", "64"),
        ("70 - 7", "63"),
        ("32 + 45", "77"),
        ("14 * 9", "126"),
    ]
    records: list[dict[str, str]] = []
    for expr, answer in examples:
        for template in templates:
            add_record(
                records,
                prompt=template.format(expr=expr),
                response=answer,
                category="digits_only",
            )
    return records


def build_yes_no_records() -> list[dict[str, str]]:
    templates = [
        "Answer only yes or no in lowercase. {question}",
        "Reply with yes or no only. {question}",
        "Use lowercase yes or no and nothing else. {question}",
    ]
    examples = [
        ("Is 14 an even number?", "yes"),
        ("Is 21 less than 19?", "no"),
        ("Is 100 greater than 12?", "yes"),
        ("Is 17 divisible by 2?", "no"),
        ("Is Warsaw in Poland?", "yes"),
        ("Is the Pacific Ocean smaller than a lake?", "no"),
        ("Is 64 equal to 8 squared?", "yes"),
        ("Is 55 a prime number?", "no"),
        ("Is water wet?", "yes"),
        ("Is a triangle a four-sided shape?", "no"),
        ("Is 1 greater than 10?", "no"),
        ("Is 0 less than 5?", "yes"),
        ("Is Python a snake and a programming language?", "yes"),
        ("Is snow usually hot?", "no"),
        ("Is 3 multiplied by 3 equal to 9?", "yes"),
        ("Is Friday before Thursday?", "no"),
        ("Is Saturn a planet?", "yes"),
        ("Is 50 smaller than 5?", "no"),
        ("Is a banana blue by default?", "no"),
        ("Is June after May?", "yes"),
        ("Is ten plus two equal to twelve?", "yes"),
        ("Is the moon made of bread?", "no"),
        ("Is autumn a season?", "yes"),
        ("Is 99 an even number?", "no"),
    ]
    records: list[dict[str, str]] = []
    for question, answer in examples:
        for template in templates:
            add_record(
                records,
                prompt=template.format(question=question),
                response=answer,
                category="yes_no",
            )
    return records


def build_one_word_records() -> list[dict[str, str]]:
    templates = [
        "Answer with one lowercase word only. {clue}",
        "Reply using exactly one lowercase word. {clue}",
    ]
    examples = [
        ("A red round fruit often used in pies.", "apple"),
        ("A large animal with a trunk.", "elephant"),
        ("The star at the center of our solar system.", "sun"),
        ("Frozen water falling from clouds.", "snow"),
        ("A vehicle with two wheels powered by pedaling.", "bicycle"),
        ("A yellow long fruit peeled before eating.", "banana"),
        ("The natural satellite of Earth.", "moon"),
        ("A domesticated animal that barks.", "dog"),
        ("A color made by mixing blue and yellow.", "green"),
        ("The day after Friday.", "saturday"),
        ("A flying vehicle used for air travel.", "airplane"),
        ("A tool used to write with ink.", "pen"),
        ("A planet known for its rings.", "saturn"),
        ("The season after summer.", "autumn"),
        ("A baby cat.", "kitten"),
        ("The opposite of cold.", "hot"),
        ("A place where books are borrowed.", "library"),
        ("A large body of salt water.", "ocean"),
        ("A scientist who studies stars and planets.", "astronomer"),
        ("A building where planes land.", "airport"),
        ("A person who writes software.", "programmer"),
        ("The color of grass.", "green"),
        ("A device used to call people.", "phone"),
        ("The meal eaten in the morning.", "breakfast"),
    ]
    records: list[dict[str, str]] = []
    for clue, answer in examples:
        for template in templates:
            add_record(
                records,
                prompt=template.format(clue=clue),
                response=answer,
                category="one_word",
            )
    return records


def build_three_word_records() -> list[dict[str, str]]:
    templates = [
        "Answer with exactly three lowercase words and nothing else. Give a short status for {topic}.",
        "Use exactly three lowercase words. Summarize {topic}.",
    ]
    examples = [
        ("project status", "project is healthy"),
        ("server health", "server looks stable"),
        ("deployment risk", "risk remains moderate"),
        ("training quality", "loss is falling"),
        ("benchmark progress", "benchmark keeps moving"),
        ("team morale", "team needs coffee"),
        ("feature scope", "scope feels contained"),
        ("next action", "run smaller smoke"),
        ("model speed", "gemma feels quicker"),
        ("controller effect", "controller changes tone"),
        ("validation trend", "validation still matters"),
        ("debugging advice", "read first failure"),
        ("release readiness", "release needs review"),
        ("incident response", "triage starts now"),
        ("code review", "review found issues"),
        ("memory usage", "memory stays moderate"),
        ("test status", "tests remain green"),
        ("api latency", "latency feels lower"),
        ("dataset quality", "dataset looks cleaner"),
        ("prompt format", "format stays strict"),
        ("training plan", "train compare iterate"),
        ("branch status", "branch stays clean"),
        ("cache behavior", "cache helps latency"),
        ("throughput estimate", "throughput seems steady"),
    ]
    records: list[dict[str, str]] = []
    for topic, answer in examples:
        for template in templates:
            add_record(
                records,
                prompt=template.format(topic=topic),
                response=answer,
                category="three_words",
            )
    return records


def build_lowercase_rewrite_records() -> list[dict[str, str]]:
    templates = [
        "Rewrite the text using lowercase only and no punctuation. Text: {text}",
        "Return the same content in lowercase words only with no punctuation: {text}",
    ]
    examples = [
        ("FAST BUILD TODAY", "fast build today"),
        ("SERVER READY NOW", "server ready now"),
        ("PLEASE CHECK LOGS", "please check logs"),
        ("NEED MORE TESTS", "need more tests"),
        ("MODEL FEELS FASTER", "model feels faster"),
        ("KEEP OUTPUT SHORT", "keep output short"),
        ("VALIDATION STILL MATTERS", "validation still matters"),
        ("READ THE FIRST ERROR", "read the first error"),
        ("USE SMALLER BATCHES", "use smaller batches"),
        ("COMPARE BASE AND CONTROLLER", "compare base and controller"),
        ("SHIPPING NEEDS DISCIPLINE", "shipping needs discipline"),
        ("TRACK THE BEST EPOCH", "track the best epoch"),
        ("CACHE REDUCES LATENCY", "cache reduces latency"),
        ("BENCHMARKS TAKE TIME", "benchmarks take time"),
        ("CLEAN DATA HELPS TRAINING", "clean data helps training"),
        ("WRITE THE SUMMARY FILE", "write the summary file"),
    ]
    records: list[dict[str, str]] = []
    for text, answer in examples:
        for template in templates:
            add_record(
                records,
                prompt=template.format(text=text),
                response=answer,
                category="lowercase_rewrite",
            )
    return records


def build_records() -> list[dict[str, str]]:
    records = [
        *build_digit_only_records(),
        *build_yes_no_records(),
        *build_one_word_records(),
        *build_three_word_records(),
        *build_lowercase_rewrite_records(),
    ]
    random.Random(SEED).shuffle(records)
    return records


def main() -> None:
    records = build_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
