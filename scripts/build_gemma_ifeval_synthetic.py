from __future__ import annotations

import json
import random
from pathlib import Path

from instruction_following_eval import evaluation_lib


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "gemma_ifeval_synthetic_en.jsonl"
)
SEED = 41

TOPICS = [
    "release planning",
    "dataset review",
    "incident response",
    "model evaluation",
    "deployment safety",
    "debugging workflow",
    "support handoff",
    "training hygiene",
    "benchmark analysis",
    "product research",
    "documentation cleanup",
    "migration planning",
    "feature scoping",
    "quality review",
    "experiment tracking",
    "latency budget",
    "customer escalation",
    "privacy review",
    "metrics cleanup",
    "launch readiness",
    "model rollout",
    "failure analysis",
    "handover checklist",
    "prompt audit",
    "service recovery",
    "capacity planning",
    "eval calibration",
    "security patch",
    "data retention",
    "cost review",
    "support routing",
    "release rollback",
]

KEYWORDS = [
    "latency",
    "coverage",
    "rollback",
    "dataset",
    "strict",
    "summary",
    "impact",
    "triage",
    "owner",
    "signal",
    "metric",
    "sample",
    "health",
    "alert",
    "stability",
    "repro",
    "logs",
    "patch",
    "handoff",
    "queue",
    "context",
    "checkpoint",
    "validation",
    "patience",
    "prompt",
    "loose",
    "failure",
    "research",
    "customer",
    "evidence",
    "guide",
    "section",
    "review",
    "migration",
    "risk",
    "window",
]

KEYWORD_GROUPS = [
    (
        KEYWORDS[(idx * 3) % len(KEYWORDS)],
        KEYWORDS[(idx * 3 + 1) % len(KEYWORDS)],
        KEYWORDS[(idx * 3 + 2) % len(KEYWORDS)],
    )
    for idx in range(len(TOPICS))
]


def add_record(
    records: list[dict],
    *,
    prompt: str,
    response: str,
    category: str,
    instruction_id_list: list[str],
    kwargs: list[dict],
) -> None:
    records.append(
        {
            "prompt": prompt,
            "response": response,
            "category": category,
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
        }
    )


def words(topic: str, count: int, *, keyword: str | None = None) -> str:
    base = [
        "clear",
        "notes",
        "help",
        "teams",
        "compare",
        "signals",
        "before",
        "acting",
        "small",
        "checks",
        "keep",
        "work",
        "steady",
        "practical",
        "review",
        "turns",
        "noise",
        "into",
        "useful",
        "next",
        "steps",
        "for",
        topic.replace(" ", "_"),
    ]
    if keyword is not None:
        base.extend([keyword, keyword, keyword])
    result: list[str] = []
    while len(result) < count:
        result.extend(base)
    return " ".join(result[:count])


def paragraph(topic: str, *, keyword: str | None = None) -> str:
    keyword_prefix = "" if keyword is None else f"{keyword} "
    return (
        f"{topic} works best when the team keeps plain notes and checks the first "
        f"failure before changing the plan. {keyword_prefix}{words(topic, 18)}."
    )


def no_comma_text(topic: str, *, keyword: str | None = None, word_count: int = 90) -> str:
    return words(topic, word_count, keyword=keyword).replace(",", "")


def build_bullet_records() -> list[dict]:
    records: list[dict] = []
    for idx, topic in enumerate(TOPICS):
        count = 3 + idx % 4
        bullets = [
            f"* {topic} item {bullet_idx + 1} keeps the plan clear"
            for bullet_idx in range(count)
        ]
        prompt = (
            f"Write advice about {topic}. Your answer must contain exactly {count} "
            "bullet points in markdown using '* '."
        )
        add_record(
            records,
            prompt=prompt,
            response="\n".join(bullets),
            category="ifeval_bullets",
            instruction_id_list=["detectable_format:number_bullet_lists"],
            kwargs=[{"num_bullets": count}],
        )
    return records


def build_json_records() -> list[dict]:
    records: list[dict] = []
    for topic, (keyword_a, keyword_b, _) in zip(TOPICS, KEYWORD_GROUPS):
        forbidden = "draft"
        prompt = (
            f"Give a JSON status object about {topic}. Include the keywords "
            f"{keyword_a} and {keyword_b}. Do not use the word {forbidden}."
        )
        response = json.dumps(
            {
                "topic": topic,
                "status": f"{keyword_a} and {keyword_b} are tracked",
                "next_step": "review the result",
            },
            ensure_ascii=False,
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_json_keywords",
            instruction_id_list=[
                "detectable_format:json_format",
                "keywords:existence",
                "keywords:forbidden_words",
            ],
            kwargs=[
                {},
                {"keywords": [keyword_a, keyword_b]},
                {"forbidden_words": [forbidden]},
            ],
        )
    return records


def build_no_comma_highlight_records() -> list[dict]:
    records: list[dict] = []
    for idx, topic in enumerate(TOPICS):
        highlight_count = 2 + idx % 4
        highlights = " ".join(
            f"*{topic.replace(' ', '_')}_{num}*" for num in range(1, highlight_count + 1)
        )
        response = f"{highlights} {no_comma_text(topic, word_count=85)}"
        prompt = (
            f"Write at least 80 words about {topic}. Do not use commas. "
            f"Highlight at least {highlight_count} sections with markdown asterisks."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_no_comma_highlights_words",
            instruction_id_list=[
                "punctuation:no_comma",
                "detectable_format:number_highlighted_sections",
                "length_constraints:number_words",
            ],
            kwargs=[
                {},
                {"num_highlights": highlight_count},
                {"relation": "at least", "num_words": 80},
            ],
        )
    return records


def build_repeat_title_records() -> list[dict]:
    records: list[dict] = []
    for topic, (keyword_a, keyword_b, _) in zip(TOPICS, KEYWORD_GROUPS):
        request = (
            f"Write a short note about {topic} with a title and the words "
            f"{keyword_a} and {keyword_b}."
        )
        prompt = (
            f"{request}\nFirst repeat the request word for word without change, "
            "then give your answer."
        )
        response = (
            f"{request}\n<<{topic.title()}>>\n"
            f"The {keyword_a} signal and {keyword_b} review keep the note useful."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_repeat_title_keywords",
            instruction_id_list=[
                "combination:repeat_prompt",
                "detectable_format:title",
                "keywords:existence",
            ],
            kwargs=[
                {"prompt_to_repeat": request},
                {},
                {"keywords": [keyword_a, keyword_b]},
            ],
        )
    return records


def build_paragraph_records() -> list[dict]:
    records: list[dict] = []
    first_words = ["anchor", "beacon", "circle", "delta"]
    for idx, topic in enumerate(TOPICS):
        count = 2 + idx % 4
        parts = [paragraph(f"{topic} part {num}") for num in range(1, count + 1)]
        prompt = (
            f"Write about {topic} in exactly {count} paragraphs. Separate paragraphs "
            "with the markdown divider ***."
        )
        add_record(
            records,
            prompt=prompt,
            response="\n***\n".join(parts),
            category="ifeval_paragraph_divider",
            instruction_id_list=["length_constraints:number_paragraphs"],
            kwargs=[{"num_paragraphs": count}],
        )

        nth = 2 if count >= 2 else 1
        first_word = first_words[idx % len(first_words)]
        first_word_parts = [
            f"{first_word} {paragraph(f'{topic} paragraph {num}')}"
            if num == nth
            else paragraph(f"{topic} paragraph {num}")
            for num in range(1, count + 1)
        ]
        prompt = (
            f"Write exactly {count} paragraphs about {topic} separated by blank lines. "
            f"Paragraph {nth} must start with the word {first_word}."
        )
        add_record(
            records,
            prompt=prompt,
            response="\n\n".join(first_word_parts),
            category="ifeval_nth_paragraph_first_word",
            instruction_id_list=["length_constraints:nth_paragraph_first_word"],
            kwargs=[
                {
                    "num_paragraphs": count,
                    "nth_paragraph": nth,
                    "first_word": first_word,
                }
            ],
        )
    return records


def build_section_case_records() -> list[dict]:
    records: list[dict] = []
    for idx, topic in enumerate(TOPICS):
        count = 2 + idx % 4
        splitter = "SECTION" if idx % 2 == 0 else "Section"
        sections = [
            f"{splitter} {num}\n{paragraph(f'{topic} section {num}')}"
            for num in range(1, count + 1)
        ]
        prompt = (
            f"Write about {topic}. Your response must have {count} sections. "
            f"Mark each section with {splitter} X."
        )
        add_record(
            records,
            prompt=prompt,
            response="\n".join(sections),
            category="ifeval_multiple_sections",
            instruction_id_list=["detectable_format:multiple_sections"],
            kwargs=[{"section_spliter": splitter, "num_sections": count}],
        )

        upper_response = "\n".join(sections).upper()
        prompt = (
            f"Write about {topic} in all capital letters. Your response must have "
            f"{count} sections marked with SECTION X."
        )
        add_record(
            records,
            prompt=prompt,
            response=upper_response,
            category="ifeval_uppercase_sections",
            instruction_id_list=[
                "change_case:english_capital",
                "detectable_format:multiple_sections",
            ],
            kwargs=[{}, {"section_spliter": "SECTION", "num_sections": count}],
        )
    return records


def build_keyword_frequency_records() -> list[dict]:
    records: list[dict] = []
    for idx, (topic, (keyword_a, keyword_b, keyword_c)) in enumerate(
        zip(TOPICS, KEYWORD_GROUPS)
    ):
        frequency = 3 + idx % 4
        response = (
            " ".join([keyword_a] * frequency)
            + " "
            + paragraph(topic, keyword=keyword_b)
        )
        prompt = (
            f"Write about {topic}. The word {keyword_a} must appear at least "
            f"{frequency} times. Include the keyword {keyword_b}."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_keyword_frequency",
            instruction_id_list=["keywords:frequency", "keywords:existence"],
            kwargs=[
                {
                    "relation": "at least",
                    "keyword": keyword_a,
                    "frequency": frequency,
                },
                {"keywords": [keyword_b]},
            ],
        )

        prompt = (
            f"Write about {topic}. Do not include the word {keyword_c}. "
            f"Make the letter z appear less than 2 times."
        )
        response = (
            f"{topic} work keeps plain notes and checks the first issue before "
            "the team changes the plan. clear notes help teams compare signals "
            "and choose a practical next step."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_forbidden_letter",
            instruction_id_list=[
                "keywords:forbidden_words",
                "keywords:letter_frequency",
            ],
            kwargs=[
                {"forbidden_words": [keyword_c]},
                {"let_relation": "less than", "letter": "z", "let_frequency": 2},
            ],
        )
    return records


def build_long_constraint_records() -> list[dict]:
    records: list[dict] = []
    word_counts = [300, 400, 500, 600]
    for idx, (topic, (keyword_a, keyword_b, _)) in enumerate(
        zip(TOPICS, KEYWORD_GROUPS)
    ):
        word_count = word_counts[idx % len(word_counts)]
        highlight_count = 3 + idx % 4
        highlights = " ".join(
            f"*{topic.replace(' ', '_')}_long_{num}*"
            for num in range(1, highlight_count + 1)
        )
        response = (
            f"{highlights} "
            f"{no_comma_text(topic, keyword=keyword_a, word_count=word_count)} "
            f"{keyword_b}"
        )
        prompt = (
            f"Write at least {word_count} words about {topic}. Do not use commas. "
            f"Include the keywords {keyword_a} and {keyword_b}. Highlight at least "
            f"{highlight_count} sections with markdown asterisks."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_long_no_comma_keywords_highlights",
            instruction_id_list=[
                "keywords:existence",
                "length_constraints:number_words",
                "punctuation:no_comma",
                "detectable_format:number_highlighted_sections",
            ],
            kwargs=[
                {"keywords": [keyword_a, keyword_b]},
                {"relation": "at least", "num_words": word_count},
                {},
                {"num_highlights": highlight_count},
            ],
        )

        frequency = 3 + idx % 4
        title_response = (
            f"<<{topic.title()}>>\n"
            + " ".join([keyword_a] * frequency)
            + " "
            + words(topic, word_count, keyword=keyword_b)
        )
        prompt = (
            f"Write at least {word_count} words about {topic}. Include a title in "
            f"double angular brackets. The word {keyword_a} must appear at least "
            f"{frequency} times. Include the keyword {keyword_b}."
        )
        add_record(
            records,
            prompt=prompt,
            response=title_response,
            category="ifeval_long_title_keyword_frequency",
            instruction_id_list=[
                "length_constraints:number_words",
                "detectable_format:title",
                "keywords:frequency",
                "keywords:existence",
            ],
            kwargs=[
                {"relation": "at least", "num_words": word_count},
                {},
                {
                    "relation": "at least",
                    "keyword": keyword_a,
                    "frequency": frequency,
                },
                {"keywords": [keyword_b]},
            ],
        )
    return records


def build_start_end_records() -> list[dict]:
    records: list[dict] = []
    endings = [
        "Is there anything else I can help with?",
        "Any other questions?",
        "That is all you need!",
    ]
    for idx, topic in enumerate(TOPICS):
        ending = endings[idx % len(endings)]
        prompt = f"Write a short answer about {topic}. End with the exact phrase: {ending}"
        response = f"{paragraph(topic)} {ending}"
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_end_phrase",
            instruction_id_list=["startend:end_checker"],
            kwargs=[{"end_phrase": ending}],
        )

        quoted = f'"<<{topic.title()}>>\n{paragraph(topic)}"'
        prompt = (
            f"Write about {topic}. Wrap the entire response in double quotation "
            "marks and include a title in double angular brackets."
        )
        add_record(
            records,
            prompt=prompt,
            response=quoted,
            category="ifeval_quote_title",
            instruction_id_list=["startend:quotation", "detectable_format:title"],
            kwargs=[{}, {}],
        )
    return records


def build_placeholder_postscript_records() -> list[dict]:
    records: list[dict] = []
    placeholders = [
        "[name]",
        "[email]",
        "[phone]",
        "[date]",
        "[role]",
        "[team]",
        "[goal]",
        "[risk]",
        "[owner]",
        "[deadline]",
    ]
    for idx, topic in enumerate(TOPICS):
        marker = "P.P.S" if idx % 2 else "P.S."
        prompt = (
            f"Write a template about {topic}. Include at least 8 square bracket "
            f"placeholders. Add a postscript starting with {marker}."
        )
        response = "\n".join(placeholders) + f"\n{marker} review before sending."
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_placeholders_postscript",
            instruction_id_list=[
                "detectable_content:number_placeholders",
                "detectable_content:postscript",
            ],
            kwargs=[
                {"num_placeholders": 8},
                {"postscript_marker": marker},
            ],
        )
    return records


def build_misc_records() -> list[dict]:
    records: list[dict] = []
    for idx, topic in enumerate(TOPICS):
        prompt = f"Answer the question about {topic} using one allowed phrase only."
        response = ["My answer is yes.", "My answer is no.", "My answer is maybe."][
            idx % 3
        ]
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_constrained_response",
            instruction_id_list=["detectable_format:constrained_response"],
            kwargs=[{}],
        )

        response = (
            f"<<{topic.title()}>>\n"
            f"first response about {topic} stays short.\n"
            "******\n"
            f"second response about {topic} takes another angle."
        )
        prompt = (
            f"Give two different responses about {topic} separated by 6 asterisk "
            "symbols and include a title."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_two_responses_title",
            instruction_id_list=["combination:two_responses", "detectable_format:title"],
            kwargs=[{}, {}],
        )

        response = no_comma_text(topic, word_count=35)
        prompt = (
            f"Write fewer than 45 words about {topic}. Do not use commas and use "
            "lowercase letters only."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="ifeval_short_lower_no_comma",
            instruction_id_list=[
                "length_constraints:number_words",
                "punctuation:no_comma",
                "change_case:english_lowercase",
            ],
            kwargs=[
                {"relation": "less than", "num_words": 45},
                {},
                {},
            ],
        )
    return records


def build_records() -> list[dict]:
    records = [
        *build_bullet_records(),
        *build_json_records(),
        *build_no_comma_highlight_records(),
        *build_repeat_title_records(),
        *build_paragraph_records(),
        *build_section_case_records(),
        *build_keyword_frequency_records(),
        *build_long_constraint_records(),
        *build_start_end_records(),
        *build_placeholder_postscript_records(),
        *build_misc_records(),
    ]
    random.Random(SEED).shuffle(records)
    return records


def validate_records(records: list[dict]) -> None:
    for idx, record in enumerate(records):
        example = evaluation_lib.InputExample(
            key=idx,
            instruction_id_list=record["instruction_id_list"],
            prompt=record["prompt"],
            kwargs=record["kwargs"],
        )
        output = evaluation_lib.test_instruction_following_strict(
            example,
            {record["prompt"]: record["response"]},
        )
        if not output.follow_all_instructions:
            failed = [
                instruction_id
                for instruction_id, followed in zip(
                    output.instruction_id_list, output.follow_instruction_list
                )
                if not followed
            ]
            raise AssertionError(
                f"Record {idx} failed {failed}: {record['prompt']!r} -> {record['response']!r}"
            )


def main() -> None:
    records = build_records()
    validate_records(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
