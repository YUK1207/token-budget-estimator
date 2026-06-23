#!/usr/bin/env python3
"""Estimate full agent-task token usage before execution."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


TASK_PROFILES = {
    "simple_question": {"complexity": 1, "searches": (0, 1), "files": (0, 1), "commands": (0, 0), "failures": (0, 0)},
    "code_explanation": {"complexity": 2, "searches": (2, 6), "files": (2, 8), "commands": (0, 1), "failures": (0, 0)},
    "small_code_change": {"complexity": 3, "searches": (3, 8), "files": (3, 10), "commands": (1, 3), "failures": (0, 1)},
    "debugging": {"complexity": 4, "searches": (5, 14), "files": (6, 20), "commands": (2, 6), "failures": (1, 4)},
    "test_repair": {"complexity": 4, "searches": (4, 12), "files": (5, 18), "commands": (3, 8), "failures": (1, 5)},
    "feature_development": {"complexity": 5, "searches": (6, 16), "files": (8, 24), "commands": (2, 6), "failures": (1, 3)},
    "frontend_build": {"complexity": 6, "searches": (6, 18), "files": (8, 28), "commands": (4, 10), "failures": (1, 4)},
    "git_publish": {"complexity": 3, "searches": (1, 4), "files": (2, 8), "commands": (4, 10), "failures": (0, 2)},
    "refactor": {"complexity": 7, "searches": (10, 28), "files": (18, 60), "commands": (4, 12), "failures": (2, 6)},
    "research_or_docs": {"complexity": 6, "searches": (6, 18), "files": (0, 8), "commands": (0, 3), "failures": (0, 2)},
    "large_project_task": {"complexity": 10, "searches": (18, 50), "files": (30, 100), "commands": (8, 20), "failures": (3, 10)},
}


@dataclass
class RepoSignals:
    file_count: int
    code_file_count: int
    test_file_count: int
    config_file_count: int
    repo_size: str


@dataclass
class CostEstimate:
    low: float
    high: float
    currency: str


@dataclass
class ProjectConfig:
    budget_tokens: int | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    currency: str = "USD"


@dataclass
class Estimate:
    task_type: str
    complexity: int
    risk: str
    confidence: str
    low_tokens: int
    high_tokens: int
    midpoint_tokens: int
    input_low_tokens: int
    input_high_tokens: int
    output_low_tokens: int
    output_high_tokens: int
    calibration_ratio: float
    repo: RepoSignals
    drivers: list[str]
    recommendation: str
    split_plan: list[str]
    context_window: int | None = None
    context_fit: str | None = None
    cost: CostEstimate | None = None
    rank: int | None = None
    budget_tokens: int | None = None
    budget_status: str | None = None


CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".scala", ".sh", ".ps1",
    ".html", ".css", ".scss", ".vue", ".svelte", ".sql",
}

CONFIG_NAMES = {
    "package.json", "pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml",
    "pom.xml", "build.gradle", "tsconfig.json", "vite.config.ts", "next.config.js",
    "pytest.ini", "tox.ini", "Dockerfile", "docker-compose.yml",
}


def load_project_config(cwd: str | Path) -> ProjectConfig:
    root = Path(cwd)
    if root.is_file():
        root = root.parent
    candidates = [root, *root.parents]
    for directory in candidates:
        path = directory / ".token-budget.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProjectConfig(
            budget_tokens=_optional_int(data.get("budget_tokens")),
            context_tokens=_optional_int(data.get("context_tokens")),
            context_window=_optional_int(data.get("context_window")),
            input_price_per_million=_optional_float(data.get("input_price_per_million")),
            output_price_per_million=_optional_float(data.get("output_price_per_million")),
            currency=str(data.get("currency", "USD")),
        )
    return ProjectConfig()


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def approx_tokens(text: str) -> int:
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return math.ceil(ascii_chars / 4 + non_ascii_chars / 1.8)


def collect_repo_signals(cwd: str, max_files: int = 12000) -> RepoSignals:
    root = Path(cwd)
    file_count = code_file_count = test_file_count = config_file_count = 0
    ignored = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", "target"}

    if not root.exists() or not root.is_dir():
        return RepoSignals(0, 0, 0, 0, "unknown")

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".cache")]
        for filename in filenames:
            file_count += 1
            path = Path(dirpath) / filename
            suffix = path.suffix.lower()
            lowered = str(path).lower()
            if suffix in CODE_EXTENSIONS:
                code_file_count += 1
            if "test" in lowered or "spec" in lowered:
                test_file_count += 1
            if filename in CONFIG_NAMES or suffix in {".toml", ".yaml", ".yml", ".json", ".ini"}:
                config_file_count += 1
            if file_count >= max_files:
                break
        if file_count >= max_files:
            break

    if code_file_count < 50:
        repo_size = "small"
    elif code_file_count < 500:
        repo_size = "medium"
    elif code_file_count < 2000:
        repo_size = "large"
    else:
        repo_size = "very_large"

    return RepoSignals(file_count, code_file_count, test_file_count, config_file_count, repo_size)


def classify_task(task: str) -> str:
    t = task.lower()
    has_git_publish = any(w in t for w in [
        "upload to github", "publish to github", "push to github", "create github repo",
        "github repo", "git remote", "git push", "open source", "release repo",
        "publish skill", "upload skill", "push the branch", "create or configure remote",
    ])
    has_code = any(w in t for w in ["code", "function", "class", "repo", "project", "代码", "函数", "项目"])
    has_test = any(w in t for w in ["test", "pytest", "jest", "ci", "failing", "failure", "测试", "报错"])
    has_bug = any(w in t for w in ["bug", "debug", "fix", "error", "exception", "traceback", "修复", "错误", "失败"])
    has_frontend = any(w in t for w in ["frontend", "ui", "react", "vue", "page", "dashboard", "browser", "前端", "页面"])
    has_refactor = any(w in t for w in ["refactor", "rewrite", "restructure", "重构", "改造"])
    has_research = any(w in t for w in ["research", "search", "look up", "调研", "搜索", "查一下"])
    has_feature = any(w in t for w in ["add", "build", "create", "implement", "feature", "新增", "实现", "开发", "生成"])
    broad = any(w in t for w in ["entire", "all", "full", "complete", "全量", "整个", "全部", "完整"])

    if has_git_publish:
        return "git_publish"
    if broad and (has_code or has_feature or has_refactor):
        return "large_project_task"
    if has_refactor:
        return "refactor"
    if has_frontend and has_feature:
        return "frontend_build"
    if has_test and has_bug:
        return "test_repair"
    if has_bug:
        return "debugging"
    if has_research:
        return "research_or_docs"
    if has_feature and has_code:
        return "feature_development"
    if has_code:
        return "code_explanation"
    if has_feature:
        return "small_code_change"
    return "simple_question"


def load_calibration(history_path: Path, task_type: str) -> tuple[float, int]:
    if not history_path.exists():
        return 1.0, 0
    ratios: list[float] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("task_type") != task_type:
            continue
        predicted = row.get("predicted_midpoint_tokens")
        actual = row.get("actual_tokens")
        if isinstance(predicted, (int, float)) and predicted > 0 and isinstance(actual, (int, float)) and actual > 0:
            ratios.append(float(actual) / float(predicted))
    if not ratios:
        return 1.0, 0
    return max(0.45, min(2.5, median(ratios))), len(ratios)


def risk_for(high_tokens: int) -> str:
    if high_tokens < 15_000:
        return "low"
    if high_tokens < 50_000:
        return "medium"
    if high_tokens < 120_000:
        return "high"
    return "extreme"


def split_plan_for(risk: str, task_type: str) -> list[str]:
    if risk in {"low", "medium"}:
        return []
    if task_type == "git_publish":
        return [
            "Inspect only the target skill folder and confirm no unrelated files will be published.",
            "Initialize or verify git state, then make one focused commit.",
            "Create or configure the GitHub remote and push after the commit is clean.",
        ]
    if task_type in {"debugging", "test_repair"}:
        return [
            "Run a discovery pass to identify the failing command, relevant files, and shortest reproduction.",
            "Fix one failure chain and run only the focused test target.",
            "Expand verification only after the focused fix passes.",
        ]
    if task_type in {"feature_development", "frontend_build", "refactor", "large_project_task"}:
        return [
            "Run a discovery pass to map touched files and existing patterns.",
            "Implement one bounded component or module.",
            "Run focused verification before broader checks.",
        ]
    return [
        "Run a discovery-only pass first.",
        "Confirm scope after the discovery result.",
        "Execute the narrowed task in a second pass.",
    ]


def budget_status_for(high_tokens: int, budget_tokens: int | None) -> str | None:
    if budget_tokens is None:
        return None
    if high_tokens <= budget_tokens:
        return "within_budget"
    if high_tokens <= budget_tokens * 1.25:
        return "near_budget"
    return "over_budget"


def context_fit_for(high_tokens: int, context_window: int | None) -> str | None:
    if context_window is None:
        return None
    if high_tokens <= context_window * 0.75:
        return "fits"
    if high_tokens <= context_window:
        return "tight"
    return "likely_exceeds"


def cost_for(
    input_low_tokens: int,
    input_high_tokens: int,
    output_low_tokens: int,
    output_high_tokens: int,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    currency: str,
) -> CostEstimate | None:
    if input_price_per_million is None or output_price_per_million is None:
        return None
    low = (input_low_tokens / 1_000_000 * input_price_per_million) + (
        output_low_tokens / 1_000_000 * output_price_per_million
    )
    high = (input_high_tokens / 1_000_000 * input_price_per_million) + (
        output_high_tokens / 1_000_000 * output_price_per_million
    )
    return CostEstimate(round(low, 4), round(high, 4), currency)


def recommendation_for(risk: str, task_type: str, budget_status: str | None = None) -> str:
    if budget_status == "over_budget":
        return "Split the task before execution because the estimate exceeds the provided budget."
    if budget_status == "near_budget":
        return "Proceed only with a tight scope because the upper estimate is close to the provided budget."
    if risk == "low":
        return "Continue after the compact budget report."
    if risk == "medium":
        return "Continue, but cap noisy command output and keep the work focused."
    if risk == "high":
        if task_type == "git_publish":
            return "Split into inspect, commit, and push phases if repository setup is uncertain."
        if task_type in {"debugging", "test_repair"}:
            return "Split into a discovery pass, then fix one reproducible failure chain."
        return "Split into discovery and implementation passes before broad edits."
    return "Start with a discovery-only pass that identifies files, commands, and risks."


def estimate(
    task: str,
    cwd: str,
    context_tokens: int,
    history_path: Path,
    budget_tokens: int | None = None,
    context_window: int | None = None,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
    currency: str = "USD",
) -> Estimate:
    repo = collect_repo_signals(cwd)
    task_type = classify_task(task)
    profile = dict(TASK_PROFILES[task_type])
    complexity = int(profile["complexity"])

    if repo.repo_size == "large":
        complexity += 1
    elif repo.repo_size == "very_large":
        complexity += 2
    if repo.test_file_count > 0 and task_type in {"debugging", "test_repair", "feature_development", "refactor"}:
        complexity += 1
    if any(w in task.lower() for w in ["all", "entire", "full", "complete", "全部", "整个", "完整"]):
        complexity += 1
    complexity = max(1, min(10, complexity))

    prompt_low = approx_tokens(task)
    prompt_high = math.ceil(prompt_low * 1.25)
    searches = profile["searches"]
    files = profile["files"]
    commands = profile["commands"]
    failures = profile["failures"]

    repo_multiplier = {"unknown": 1.0, "small": 0.9, "medium": 1.0, "large": 1.25, "very_large": 1.55}[repo.repo_size]

    low_input = (
        prompt_low
        + context_tokens
        + searches[0] * 250
        + files[0] * 700 * repo_multiplier
        + commands[0] * 400
        + failures[0] * 800
    )
    low_output = complexity * 700 + 300
    high_input = (
        prompt_high
        + context_tokens
        + searches[1] * 900
        + files[1] * 2500 * repo_multiplier
        + commands[1] * 2500
        + failures[1] * 6000
    )
    high_output = complexity * 2200 + 2200
    low = low_input + low_output
    high = high_input + high_output

    ratio, sample_count = load_calibration(history_path, task_type)
    low_factor = 0.85 if sample_count < 5 else max(0.65, ratio * 0.85)
    high_factor = 1.25 if sample_count < 5 else min(2.75, ratio * 1.2)
    low_input = int(low_input * low_factor)
    low_output = int(low_output * low_factor)
    high_input = int(high_input * high_factor)
    high_output = int(high_output * high_factor)
    low = low_input + low_output
    high = high_input + high_output
    midpoint = int((low + high) / 2)
    risk = risk_for(high)
    confidence = "low" if sample_count < 3 else "medium" if sample_count < 10 else "high"
    rounded_high = round_to_nice(high)
    budget_status = budget_status_for(rounded_high, budget_tokens)
    rounded_input_low = round_to_nice(low_input)
    rounded_input_high = round_to_nice(high_input)
    rounded_output_low = round_to_nice(low_output)
    rounded_output_high = round_to_nice(high_output)
    context_fit = context_fit_for(rounded_high, context_window)
    cost = cost_for(
        rounded_input_low,
        rounded_input_high,
        rounded_output_low,
        rounded_output_high,
        input_price_per_million,
        output_price_per_million,
        currency,
    )

    drivers = [
        f"{searches[0]}-{searches[1]} searches",
        f"reading {files[0]}-{files[1]} files",
        f"{commands[0]}-{commands[1]} commands",
    ]
    if failures[1]:
        drivers.append(f"{failures[0]}-{failures[1]} failure/retry loops")
    if repo.repo_size in {"large", "very_large"}:
        drivers.append(f"{repo.repo_size} repository")

    return Estimate(
        task_type=task_type,
        complexity=complexity,
        risk=risk,
        confidence=confidence,
        low_tokens=round_to_nice(low),
        high_tokens=rounded_high,
        midpoint_tokens=round_to_nice(midpoint),
        input_low_tokens=rounded_input_low,
        input_high_tokens=rounded_input_high,
        output_low_tokens=rounded_output_low,
        output_high_tokens=rounded_output_high,
        calibration_ratio=round(ratio, 3),
        repo=repo,
        drivers=drivers,
        recommendation=recommendation_for(risk, task_type, budget_status),
        split_plan=split_plan_for(risk, task_type) if budget_status != "within_budget" else [],
        context_window=context_window,
        context_fit=context_fit,
        cost=cost,
        budget_tokens=budget_tokens,
        budget_status=budget_status,
    )


def round_to_nice(value: int) -> int:
    if value < 10_000:
        return int(math.ceil(value / 500) * 500)
    return int(math.ceil(value / 1000) * 1000)


def compare_tasks(
    tasks: list[str],
    cwd: str,
    context_tokens: int,
    history_path: Path,
    budget_tokens: int | None = None,
    context_window: int | None = None,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
    currency: str = "USD",
) -> list[Estimate]:
    results = [
        estimate(
            task,
            cwd,
            context_tokens,
            history_path,
            budget_tokens,
            context_window,
            input_price_per_million,
            output_price_per_million,
            currency,
        )
        for task in tasks
    ]
    results.sort(key=lambda item: item.midpoint_tokens, reverse=True)
    for idx, result in enumerate(results, start=1):
        result.rank = idx
    return results


def render_markdown(result: Estimate) -> str:
    low = f"{result.low_tokens // 1000}k" if result.low_tokens >= 1000 else str(result.low_tokens)
    high = f"{result.high_tokens // 1000}k" if result.high_tokens >= 1000 else str(result.high_tokens)
    lines = [
        "**Token Preflight**",
        f"- Estimate: {low}-{high} tokens",
        f"- Risk: {result.risk}",
        f"- Confidence: {result.confidence}",
        f"- Task type: {result.task_type}",
        f"- Complexity: {result.complexity}/10",
        f"- Input tokens: {result.input_low_tokens}-{result.input_high_tokens}",
        f"- Output tokens: {result.output_low_tokens}-{result.output_high_tokens}",
        f"- Repo signals: {result.repo.repo_size}, {result.repo.code_file_count} code files, {result.repo.test_file_count} test-like files",
        f"- Main drivers: {', '.join(result.drivers)}",
        f"- Recommendation: {result.recommendation}",
    ]
    if result.context_window is not None:
        lines.append(f"- Context fit: {result.context_fit} against {result.context_window} tokens")
    if result.cost is not None:
        lines.append(f"- Cost estimate: {result.cost.currency} {result.cost.low}-{result.cost.high}")
    if result.budget_tokens is not None:
        budget = f"{result.budget_tokens // 1000}k" if result.budget_tokens >= 1000 else str(result.budget_tokens)
        lines.append(f"- Budget check: {result.budget_status} against {budget} tokens")
    if result.split_plan:
        lines.append("- Split plan:")
        lines.extend(f"  {idx}. {step}" for idx, step in enumerate(result.split_plan, start=1))
    return "\n".join(lines)


def render_compare_markdown(results: list[Estimate]) -> str:
    lines = ["**Token Budget Comparison**", "", "| Rank | Estimate | Risk | Type | Recommendation |", "|---:|---:|---|---|---|"]
    for result in results:
        low = f"{result.low_tokens // 1000}k" if result.low_tokens >= 1000 else str(result.low_tokens)
        high = f"{result.high_tokens // 1000}k" if result.high_tokens >= 1000 else str(result.high_tokens)
        lines.append(
            f"| {result.rank} | {low}-{high} | {result.risk} | {result.task_type} | {result.recommendation} |"
        )
    return "\n".join(lines)


def record(history_path: Path, args: argparse.Namespace) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task": args.task,
        "task_type": args.task_type,
        "predicted_low_tokens": args.predicted_low,
        "predicted_high_tokens": args.predicted_high,
        "predicted_midpoint_tokens": args.predicted_midpoint,
        "actual_tokens": args.actual_tokens,
    }
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_history_path() -> Path:
    return Path.home() / ".codex" / "token-budget-estimator" / "history.jsonl"


def option_value(cli_value: object, config_value: object, default: object = None) -> object:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def load_tasks_from_args(tasks: list[str] | None, tasks_file: Path | None) -> list[str]:
    loaded: list[str] = []
    if tasks:
        loaded.extend(task for task in tasks if task.strip())
    if tasks_file is not None:
        loaded.extend(line.strip() for line in tasks_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not loaded:
        raise SystemExit("compare requires at least one --task or --tasks-file entry")
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate full agent-task token usage.")
    sub = parser.add_subparsers(dest="command", required=True)

    estimate_parser = sub.add_parser("estimate")
    estimate_parser.add_argument("--task", required=True)
    estimate_parser.add_argument("--cwd", default=os.getcwd())
    estimate_parser.add_argument("--context-tokens", type=int, default=None)
    estimate_parser.add_argument("--budget", type=int, default=None, help="Optional token budget ceiling for this task.")
    estimate_parser.add_argument("--context-window", type=int, default=None, help="Optional context window size in tokens.")
    estimate_parser.add_argument("--input-price-per-million", type=float, default=None)
    estimate_parser.add_argument("--output-price-per-million", type=float, default=None)
    estimate_parser.add_argument("--currency", default=None)
    estimate_parser.add_argument("--history", type=Path, default=default_history_path())
    estimate_parser.add_argument("--json", action="store_true")

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--task", action="append", default=None, help="Task to compare. Repeat for multiple tasks.")
    compare_parser.add_argument("--tasks-file", type=Path, default=None, help="Text file with one task per line.")
    compare_parser.add_argument("--cwd", default=os.getcwd())
    compare_parser.add_argument("--context-tokens", type=int, default=None)
    compare_parser.add_argument("--budget", type=int, default=None)
    compare_parser.add_argument("--context-window", type=int, default=None)
    compare_parser.add_argument("--input-price-per-million", type=float, default=None)
    compare_parser.add_argument("--output-price-per-million", type=float, default=None)
    compare_parser.add_argument("--currency", default=None)
    compare_parser.add_argument("--history", type=Path, default=default_history_path())
    compare_parser.add_argument("--json", action="store_true")

    record_parser = sub.add_parser("record")
    record_parser.add_argument("--task", required=True)
    record_parser.add_argument("--task-type", required=True)
    record_parser.add_argument("--predicted-low", type=int, required=True)
    record_parser.add_argument("--predicted-high", type=int, required=True)
    record_parser.add_argument("--predicted-midpoint", type=int, required=True)
    record_parser.add_argument("--actual-tokens", type=int, required=True)
    record_parser.add_argument("--history", type=Path, default=default_history_path())

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "estimate":
        config = load_project_config(args.cwd)
        result = estimate(
            args.task,
            args.cwd,
            int(option_value(args.context_tokens, config.context_tokens, 0)),
            args.history,
            option_value(args.budget, config.budget_tokens),
            option_value(args.context_window, config.context_window),
            option_value(args.input_price_per_million, config.input_price_per_million),
            option_value(args.output_price_per_million, config.output_price_per_million),
            str(option_value(args.currency, config.currency, "USD")),
        )
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print(render_markdown(result))
        return 0

    if args.command == "compare":
        config = load_project_config(args.cwd)
        tasks = load_tasks_from_args(args.task, args.tasks_file)
        results = compare_tasks(
            tasks,
            args.cwd,
            int(option_value(args.context_tokens, config.context_tokens, 0)),
            args.history,
            option_value(args.budget, config.budget_tokens),
            option_value(args.context_window, config.context_window),
            option_value(args.input_price_per_million, config.input_price_per_million),
            option_value(args.output_price_per_million, config.output_price_per_million),
            str(option_value(args.currency, config.currency, "USD")),
        )
        if args.json:
            print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
        else:
            print(render_compare_markdown(results))
        return 0

    if args.command == "record":
        record(args.history, args)
        print(f"Recorded token budget sample to {args.history}")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
