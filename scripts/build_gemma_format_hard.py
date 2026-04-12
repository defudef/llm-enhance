from __future__ import annotations

import json
import random
from pathlib import Path


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "gemma_format_hard_en.jsonl"
)
SEED = 23


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


def build_bullets_with_keywords_records() -> list[dict[str, str]]:
    scenarios = [
        ("a launch checklist", "latency", "coverage", "owners", "timing"),
        ("a release review", "quality", "rollback", "testing", "risks"),
        ("a benchmark plan", "dataset", "strict", "loose", "summary"),
        ("a debugging checklist", "logs", "repro", "signal", "fix"),
        ("a training plan", "data", "validation", "checkpoint", "patience"),
        ("a deployment brief", "traffic", "health", "alerts", "stability"),
        ("a code review summary", "regression", "tests", "diff", "scope"),
        ("an incident update", "impact", "triage", "mitigation", "followup"),
        ("a project status note", "milestone", "blocking", "owner", "nextstep"),
        ("a release invitation", "agenda", "context", "owners", "actions"),
    ]
    records: list[dict[str, str]] = []
    for topic, keyword_a, keyword_b, keyword_c, keyword_d in scenarios:
        prompt = (
            "Write exactly 3 markdown bullet points using '* '. "
            f"Use lowercase letters only and include the keywords {keyword_a} and {keyword_b}. "
            f"The topic is {topic}."
        )
        response = "\n".join(
            [
                f"* {keyword_a} guides the plan",
                f"* {keyword_b} keeps the team honest",
                f"* {keyword_c} and {keyword_d} close the loop",
            ]
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="bullets_keywords",
        )
    return records


def build_title_quote_records() -> list[dict[str, str]]:
    subjects = [
        "resignation",
        "project update",
        "benchmark summary",
        "release invitation",
        "training plan",
        "incident report",
        "migration note",
        "performance review",
        "support handoff",
        "deployment recap",
    ]
    records: list[dict[str, str]] = []
    for subject in subjects:
        prompt = (
            f'Write a short email about {subject}. '
            'Wrap the entire response in double quotation marks and include a title in double angular brackets.'
        )
        response = (
            f'"<<{subject.title()}>>\n'
            f'Dear team\n'
            f'This note covers {subject} and keeps the message brief and clear.\n'
            f'Please review the details and reply if you need changes.\n'
            f'Best regards"'
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="title_quote",
        )
    return records


def build_repeat_then_answer_records() -> list[dict[str, str]]:
    tasks = [
        "Write a polite resignation email.",
        "Write a project kickoff note.",
        "Write a benchmark update.",
        "Write a release reminder.",
        "Write a code review request.",
        "Write a training summary.",
        "Write a deployment notice.",
        "Write an incident followup.",
        "Write a migration update.",
        "Write a short apology email.",
    ]
    records: list[dict[str, str]] = []
    for task in tasks:
        prompt = (
            f"{task} First repeat the request word for word without change then write the answer."
        )
        response = (
            f"{task}\n"
            "Dear team,\n"
            "I am sharing the requested message in a direct and professional tone.\n"
            "Please review it and adjust any names or dates before sending."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="repeat_then_answer",
        )
    return records


def build_placeholders_records() -> list[dict[str, str]]:
    roles = [
        "junior analyst",
        "support specialist",
        "qa intern",
        "product assistant",
        "marketing coordinator",
        "operations trainee",
        "sales intern",
        "research assistant",
        "project coordinator",
        "technical writer",
    ]
    records: list[dict[str, str]] = []
    for role in roles:
        prompt = (
            f"Write a resume for a {role}. Include at least 12 square-bracket placeholders and keep the response in english."
        )
        response = (
            "[name]\n"
            "[address]\n"
            "[email]\n"
            "[phone]\n"
            "[summary]\n"
            "[school]\n"
            "[graduation_year]\n"
            "[skills]\n"
            "[project_one]\n"
            "[project_two]\n"
            "[experience]\n"
            "[references]"
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="placeholders",
        )
    return records


def build_case_constraint_records() -> list[dict[str, str]]:
    lowercase_topics = [
        "why testing matters",
        "how to debug faster",
        "why validation matters",
        "how to compare models",
        "how to review a diff",
        "why logs matter",
        "how to run a benchmark",
        "why checkpoints help",
        "how to keep scope small",
        "why data quality matters",
    ]
    uppercase_topics = [
        "release readiness",
        "incident severity",
        "project risks",
        "migration guidance",
        "testing strategy",
        "benchmark summary",
        "debugging discipline",
        "deployment checklist",
        "training warnings",
        "code review notes",
    ]
    records: list[dict[str, str]] = []
    for topic in lowercase_topics:
        prompt = (
            f"Write 2 short paragraphs about {topic} in english lowercase letters only. Label them paragraph 1 and paragraph 2."
        )
        response = (
            "paragraph 1\n"
            f"{topic} stays easier when the team prefers plain signals over drama.\n\n"
            "paragraph 2\n"
            "small disciplined steps usually beat noisy confidence and rushed guesses."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="lowercase_paragraphs",
        )
    for topic in uppercase_topics:
        prompt = (
            f"Write 2 short paragraphs about {topic} using all capital letters only. Label them PARAGRAPH 1 and PARAGRAPH 2."
        )
        response = (
            "PARAGRAPH 1\n"
            f"{topic.upper()} REQUIRES CLARITY AND DISCIPLINE FROM THE WHOLE TEAM.\n\n"
            "PARAGRAPH 2\n"
            "GOOD PROCESS REDUCES CHAOS AND MAKES FOLLOWUP ACTIONS EASIER TO TRACK."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="uppercase_paragraphs",
        )
    return records


def build_word_count_keyword_records() -> list[dict[str, str]]:
    themes = [
        ("benchmarking discipline", "correlated", "experiencing"),
        ("validation hygiene", "measured", "observing"),
        ("deployment safety", "prepared", "monitoring"),
        ("incident review", "contained", "tracking"),
        ("code review habits", "careful", "learning"),
        ("training quality", "signals", "improving"),
        ("debugging process", "evidence", "checking"),
        ("release readiness", "owners", "reporting"),
    ]
    records: list[dict[str, str]] = []
    for theme, keyword_a, keyword_b in themes:
        prompt = (
            f"Write at least 40 words about {theme}. "
            f"Include the keywords {keyword_a} and {keyword_b} and do not use commas."
        )
        response = (
            f"{keyword_a} teams keep improving when they stay calm and keep clear notes. "
            f"{keyword_b} teams learn faster when each change has a reason and a measurable goal. "
            "this note stays focused on practical steps and keeps the wording simple for repeatable evaluation. "
            "the plan remains short structured and easy to check."
        )
        add_record(
            records,
            prompt=prompt,
            response=response,
            category="word_count_keywords",
        )
    return records


def build_records() -> list[dict[str, str]]:
    records = [
        *build_bullets_with_keywords_records(),
        *build_title_quote_records(),
        *build_repeat_then_answer_records(),
        *build_placeholders_records(),
        *build_case_constraint_records(),
        *build_word_count_keyword_records(),
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
