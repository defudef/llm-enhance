from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "gemma_format_smoke_en.jsonl"
)


def add_record(records: list[dict[str, str]], prompt: str, response: str) -> None:
    records.append({"prompt": prompt, "response": response})


def build_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    arithmetic_examples = [
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
    ]
    for expr, answer in arithmetic_examples:
        add_record(
            records,
            f"Answer with digits only. What is {expr}?",
            answer,
        )

    yes_no_examples = [
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
    ]
    for question, answer in yes_no_examples:
        add_record(
            records,
            f"Answer only yes or no in lowercase. {question}",
            answer,
        )

    one_word_examples = [
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
    ]
    for clue, answer in one_word_examples:
        add_record(
            records,
            f"Answer with one lowercase word only. {clue}",
            answer,
        )

    three_word_examples = [
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
    ]
    for topic, answer in three_word_examples:
        add_record(
            records,
            (
                "Answer with exactly three lowercase words and nothing else. "
                f"Give a short status for {topic}."
            ),
            answer,
        )

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
