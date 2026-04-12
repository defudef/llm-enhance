from __future__ import annotations

import json
from pathlib import Path

from instruction_following_eval import evaluation_lib


def default_ifeval_input_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "instruction_following_eval"
        / "data"
        / "input_data.jsonl"
    )


def write_prompt_responses(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def build_accuracy_report(outputs: list[evaluation_lib.OutputExample]) -> dict:
    prompt_total = len(outputs)
    prompt_correct = sum(int(output.follow_all_instructions) for output in outputs)
    instruction_total = sum(len(output.follow_instruction_list) for output in outputs)
    instruction_correct = sum(
        sum(int(followed) for followed in output.follow_instruction_list)
        for output in outputs
    )

    tier0_total: dict[str, int] = {}
    tier0_correct: dict[str, int] = {}
    tier1_total: dict[str, int] = {}
    tier1_correct: dict[str, int] = {}

    for output in outputs:
        for instruction_id, followed in zip(
            output.instruction_id_list, output.follow_instruction_list
        ):
            tier1_total[instruction_id] = tier1_total.get(instruction_id, 0) + 1
            tier1_correct[instruction_id] = tier1_correct.get(instruction_id, 0) + int(
                followed
            )

            tier0_id = instruction_id.split(":")[0]
            tier0_total[tier0_id] = tier0_total.get(tier0_id, 0) + 1
            tier0_correct[tier0_id] = tier0_correct.get(tier0_id, 0) + int(followed)

    return {
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_accuracy": 0.0 if prompt_total == 0 else prompt_correct / prompt_total,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "instruction_accuracy": (
            0.0 if instruction_total == 0 else instruction_correct / instruction_total
        ),
        "tier0": {
            key: {
                "correct": tier0_correct[key],
                "total": tier0_total[key],
                "accuracy": tier0_correct[key] / tier0_total[key],
            }
            for key in sorted(tier0_total)
        },
        "tier1": {
            key: {
                "correct": tier1_correct[key],
                "total": tier1_total[key],
                "accuracy": tier1_correct[key] / tier1_total[key],
            }
            for key in sorted(tier1_total)
        },
    }


def compute_final_score(summary: dict) -> float:
    strict = summary["strict"]
    loose = summary["loose"]
    return (
        strict["prompt_accuracy"]
        + strict["instruction_accuracy"]
        + loose["prompt_accuracy"]
        + loose["instruction_accuracy"]
    ) / 4


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def build_progress_payload(
    *,
    completed: int,
    total: int,
    elapsed_seconds: float,
) -> dict:
    avg_seconds = 0.0 if completed == 0 else elapsed_seconds / completed
    remaining = max(total - completed, 0)
    eta_seconds = avg_seconds * remaining if completed > 0 else None
    return {
        "completed_examples": completed,
        "total_examples": total,
        "percent_complete": (0.0 if total == 0 else completed / total),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_duration(elapsed_seconds),
        "avg_seconds_per_example": avg_seconds,
        "avg_human_per_example": format_duration(avg_seconds),
        "eta_seconds": eta_seconds,
        "eta_human": None if eta_seconds is None else format_duration(eta_seconds),
    }


def build_progress_message(
    *,
    completed: int,
    total: int,
    elapsed_seconds: float,
    label: str,
    final_score: float | None = None,
) -> str:
    progress = build_progress_payload(
        completed=completed,
        total=total,
        elapsed_seconds=elapsed_seconds,
    )
    parts = [
        f"{label} {completed}/{total}",
        f"{progress['percent_complete'] * 100:.1f}%",
        f"elapsed {progress['elapsed_human']}",
        f"avg {progress['avg_human_per_example']}/prompt",
    ]
    if progress["eta_human"] is not None:
        parts.append(f"eta {progress['eta_human']}")
    if final_score is not None:
        parts.append(f"partial final {final_score:.4f}")
    return " | ".join(parts)


def evaluate_prompt_responses(
    inputs: list[evaluation_lib.InputExample],
    prompt_to_response: dict[str, str],
    *,
    output_dir: Path,
    strict_results_name: str = "eval_results_strict.jsonl",
    loose_results_name: str = "eval_results_loose.jsonl",
    summary_name: str = "summary.json",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    strict_outputs = [
        evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        for inp in inputs
    ]
    loose_outputs = [
        evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        for inp in inputs
    ]

    evaluation_lib.write_outputs(output_dir / strict_results_name, strict_outputs)
    evaluation_lib.write_outputs(output_dir / loose_results_name, loose_outputs)

    summary = {
        "examples": len(inputs),
        "strict": build_accuracy_report(strict_outputs),
        "loose": build_accuracy_report(loose_outputs),
    }
    write_json(output_dir / summary_name, summary)
    return summary


def persist_ifeval_progress(
    *,
    inputs: list[evaluation_lib.InputExample],
    completed: int,
    records: list[dict[str, str]],
    prompt_to_response: dict[str, str],
    output_dir: Path,
    total_examples: int,
    elapsed_seconds: float,
    write_partial_eval: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_prompt_responses(output_dir / "responses.jsonl", records)
    progress = build_progress_payload(
        completed=completed,
        total=total_examples,
        elapsed_seconds=elapsed_seconds,
    )
    if write_partial_eval and completed > 0:
        partial_summary = evaluate_prompt_responses(
            inputs[:completed],
            prompt_to_response,
            output_dir=output_dir,
            strict_results_name="eval_results_strict.partial.jsonl",
            loose_results_name="eval_results_loose.partial.jsonl",
            summary_name="summary.partial.json",
        )
        progress["partial_summary"] = partial_summary
        progress["partial_final_score"] = compute_final_score(partial_summary)
    write_json(output_dir / "progress.json", progress)
    return progress
