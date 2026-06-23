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


@dataclass
class PromptOptimization:
    mode: str
    original_task: str
    optimized_prompt: str
    original_estimate: Estimate
    optimized_estimate: Estimate
    auto_split: list[str]
    scope_guard: list[str]
    output_caps: list[str]
    savings_summary: str


@dataclass
class ROIAssessment:
    score: int
    label: str
    decision: str
    rationale: list[str]
    expected_value: str
    cost_pressure: str


@dataclass
class BudgetContract:
    mode: str
    max_files: int
    max_commands: int
    max_log_lines: int
    allowed_actions: list[str]
    denied_actions: list[str]
    stop_loss: list[str]


@dataclass
class ContextDietPlan:
    read_first: list[str]
    read_second: list[str]
    avoid_initially: list[str]
    escalation_rule: str


@dataclass
class PreflightController:
    forecast: Estimate
    roi: ROIAssessment
    contract: BudgetContract
    diet: ContextDietPlan
    recommended_first_pass_prompt: str


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


def optimization_profile(mode: str) -> dict[str, object]:
    profiles: dict[str, dict[str, object]] = {
        "cheap": {
            "label": "cheap",
            "goal": "Find the smallest safe next step that reduces uncertainty before implementation.",
            "verification": "Run only the narrowest relevant check, or report the exact check to run if unavailable.",
            "max_files": "Inspect only the top 3-6 likely relevant files.",
            "max_commands": "Run at most 1-2 focused commands.",
            "output_cap": "Keep command output summaries under 80 lines and omit unrelated logs.",
            "reduction": 0.45,
        },
        "balanced": {
            "label": "balanced",
            "goal": "Complete the requested task with scoped discovery, focused edits, and focused validation.",
            "verification": "Run focused validation first, then one broader check if the focused check passes.",
            "max_files": "Inspect only files connected to the target behavior before broadening scope.",
            "max_commands": "Run focused commands before broad test or build commands.",
            "output_cap": "Summarize command output and include only actionable failure excerpts.",
            "reduction": 0.65,
        },
        "thorough": {
            "label": "thorough",
            "goal": "Complete the task carefully while still avoiding unrelated repository exploration.",
            "verification": "Run focused validation plus broader regression checks where available.",
            "max_files": "Inspect related modules, tests, and configs before implementation.",
            "max_commands": "Run necessary test/build commands, but cap repeated failures.",
            "output_cap": "Summarize logs by failure chain and avoid dumping full repeated output.",
            "reduction": 0.82,
        },
    }
    if mode not in profiles:
        raise ValueError(f"unknown optimization mode: {mode}")
    return profiles[mode]


def choose_mode(mode: str | None, estimate_result: Estimate) -> str:
    if mode:
        return mode
    if estimate_result.risk in {"high", "extreme"} or estimate_result.budget_status == "over_budget":
        return "cheap"
    if estimate_result.risk == "medium":
        return "balanced"
    return "thorough"


def compact_task_text(task: str) -> str:
    words = task.replace("\n", " ").split()
    compacted: list[str] = []
    previous = None
    for word in words:
        lowered = word.lower()
        if lowered == previous:
            continue
        compacted.append(word)
        previous = lowered
    return " ".join(compacted).strip()


def build_scope_guard(task_type: str, profile: dict[str, object]) -> list[str]:
    guards = [
        str(profile["max_files"]),
        "Do not read unrelated directories or generated/vendor files unless they become necessary.",
        "Stop and report if the task appears broader than the stated scope.",
    ]
    if task_type in {"debugging", "test_repair"}:
        guards.append("Trace one failure chain before editing.")
    elif task_type == "git_publish":
        guards.append("Only inspect and publish the target repository files.")
    elif task_type in {"feature_development", "frontend_build", "refactor", "large_project_task"}:
        guards.append("Map existing patterns before adding new abstractions.")
    return guards


def build_output_caps(profile: dict[str, object]) -> list[str]:
    return [
        str(profile["output_cap"]),
        "For test failures, include the failing test name, first relevant stack frame, and concise diagnosis.",
        "Do not paste full build logs unless the user asks for raw output.",
    ]


def build_auto_split(original: Estimate, mode: str) -> list[str]:
    if mode == "cheap":
        if original.task_type in {"debugging", "test_repair"}:
            return [
                "Discovery: identify the focused failing command and relevant files without editing.",
                "Fix: patch one confirmed failure chain.",
                "Verify: run the focused command and summarize remaining failures.",
            ]
        if original.task_type == "git_publish":
            return [
                "Inspect publish scope and git status.",
                "Commit only intended files.",
                "Create/configure remote and push.",
            ]
        return [
            "Discovery: identify relevant files, commands, and risks.",
            "Implementation: perform the smallest scoped change.",
            "Verification: run focused checks and report residual risk.",
        ]
    if original.risk in {"high", "extreme"}:
        return split_plan_for(original.risk, original.task_type)
    return []


def build_optimized_prompt(task: str, original: Estimate, mode: str) -> tuple[str, list[str], list[str], list[str]]:
    profile = optimization_profile(mode)
    compacted = compact_task_text(task)
    scope_guard = build_scope_guard(original.task_type, profile)
    output_caps = build_output_caps(profile)
    auto_split = build_auto_split(original, mode)

    lines = [
        "# Codex Task Contract",
        "",
        "## Goal",
        str(profile["goal"]),
        "",
        "## Original Request",
        compacted,
        "",
        "## Budget Mode",
        mode,
        "",
        "## Scope",
        *[f"- {item}" for item in scope_guard],
        "",
        "## Execution Plan",
    ]
    if auto_split:
        lines.extend(f"- {item}" for item in auto_split)
    else:
        lines.append("- Execute the request in one focused pass.")
    lines.extend(
        [
            "",
            "## Output Caps",
            *[f"- {item}" for item in output_caps],
            "",
            "## Validation",
            f"- {profile['verification']}",
            "- End with changed files, checks run, and unresolved risks.",
        ]
    )
    return "\n".join(lines), auto_split, scope_guard, output_caps


def scaled_estimate(base: Estimate, reduction: float) -> Estimate:
    low_tokens = round_to_nice(int(base.low_tokens * reduction))
    high_tokens = round_to_nice(int(base.high_tokens * reduction))
    midpoint_tokens = round_to_nice(int(base.midpoint_tokens * reduction))
    scaled = Estimate(
        task_type=base.task_type,
        complexity=max(1, int(math.ceil(base.complexity * reduction))),
        risk=risk_for(high_tokens),
        confidence=base.confidence,
        low_tokens=low_tokens,
        high_tokens=high_tokens,
        midpoint_tokens=midpoint_tokens,
        input_low_tokens=round_to_nice(int(base.input_low_tokens * reduction)),
        input_high_tokens=round_to_nice(int(base.input_high_tokens * reduction)),
        output_low_tokens=round_to_nice(int(base.output_low_tokens * reduction)),
        output_high_tokens=round_to_nice(int(base.output_high_tokens * reduction)),
        calibration_ratio=base.calibration_ratio,
        repo=base.repo,
        drivers=base.drivers,
        recommendation=recommendation_for(risk_for(high_tokens), base.task_type, budget_status_for(high_tokens, base.budget_tokens)),
        split_plan=split_plan_for(risk_for(high_tokens), base.task_type),
        context_window=base.context_window,
        context_fit=context_fit_for(high_tokens, base.context_window),
        cost=cost_for(
            round_to_nice(int(base.input_low_tokens * reduction)),
            round_to_nice(int(base.input_high_tokens * reduction)),
            round_to_nice(int(base.output_low_tokens * reduction)),
            round_to_nice(int(base.output_high_tokens * reduction)),
            None,
            None,
            "USD",
        ),
        budget_tokens=base.budget_tokens,
        budget_status=budget_status_for(high_tokens, base.budget_tokens),
    )
    if base.cost is not None:
        scaled.cost = CostEstimate(round(base.cost.low * reduction, 4), round(base.cost.high * reduction, 4), base.cost.currency)
    return scaled


def optimize_task(
    task: str,
    cwd: str,
    context_tokens: int,
    history_path: Path,
    budget_tokens: int | None = None,
    context_window: int | None = None,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
    currency: str = "USD",
    mode: str | None = None,
) -> PromptOptimization:
    original = estimate(
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
    chosen_mode = choose_mode(mode, original)
    optimized_prompt, auto_split, scope_guard, output_caps = build_optimized_prompt(task, original, chosen_mode)
    reduction = float(optimization_profile(chosen_mode)["reduction"])
    optimized = scaled_estimate(original, reduction)
    saved_low = max(0, original.low_tokens - optimized.low_tokens)
    saved_high = max(0, original.high_tokens - optimized.high_tokens)
    savings_summary = f"Estimated reduction: {saved_low}-{saved_high} tokens by narrowing scope, capping output, and staging execution."
    return PromptOptimization(
        mode=chosen_mode,
        original_task=task,
        optimized_prompt=optimized_prompt,
        original_estimate=original,
        optimized_estimate=optimized,
        auto_split=auto_split,
        scope_guard=scope_guard,
        output_caps=output_caps,
        savings_summary=savings_summary,
    )


def assess_roi(forecast: Estimate) -> ROIAssessment:
    value_by_type = {
        "simple_question": 35,
        "code_explanation": 45,
        "small_code_change": 65,
        "debugging": 78,
        "test_repair": 82,
        "feature_development": 72,
        "frontend_build": 68,
        "git_publish": 62,
        "refactor": 58,
        "research_or_docs": 55,
        "large_project_task": 50,
    }
    risk_penalty = {"low": 0, "medium": 12, "high": 28, "extreme": 45}[forecast.risk]
    budget_penalty = {"within_budget": 0, "near_budget": 10, "over_budget": 25, None: 0}[forecast.budget_status]
    context_penalty = {"fits": 0, "tight": 12, "likely_exceeds": 25, None: 0}[forecast.context_fit]
    value = value_by_type.get(forecast.task_type, 50)
    score = max(1, min(100, value - risk_penalty - budget_penalty - context_penalty + min(10, forecast.complexity)))

    rationale = [
        f"Task value baseline is {value} for {forecast.task_type}.",
        f"Risk pressure is {forecast.risk}.",
    ]
    if forecast.budget_status:
        rationale.append(f"Budget status is {forecast.budget_status}.")
    if forecast.context_fit:
        rationale.append(f"Context fit is {forecast.context_fit}.")

    if score >= 70 and forecast.risk in {"low", "medium"}:
        decision = "execute_now"
        label = "high"
    elif score >= 45 and forecast.risk != "extreme":
        decision = "split_first" if forecast.budget_status == "over_budget" else "execute_now"
        label = "medium"
    elif forecast.task_type in {"debugging", "test_repair", "large_project_task", "refactor"}:
        decision = "discovery_first"
        label = "uncertain"
    else:
        decision = "defer"
        label = "low"

    if forecast.high_tokens < 50_000:
        cost_pressure = "low"
    elif forecast.high_tokens < 120_000:
        cost_pressure = "medium"
    else:
        cost_pressure = "high"

    expected_value = "high" if value >= 75 else "medium" if value >= 55 else "low"
    return ROIAssessment(score, label, decision, rationale, expected_value, cost_pressure)


def contract_limits(mode: str, forecast: Estimate) -> tuple[int, int, int]:
    if mode == "cheap":
        return (6, 2, 80)
    if mode == "balanced":
        return (14, 4, 160)
    if forecast.risk == "extreme":
        return (24, 8, 240)
    return (30, 10, 320)


def generate_budget_contract(forecast: Estimate, mode: str) -> BudgetContract:
    max_files, max_commands, max_log_lines = contract_limits(mode, forecast)
    allowed = [
        f"Read at most {max_files} relevant files before the first report.",
        f"Run at most {max_commands} focused commands before reassessing.",
        f"Summarize command output under {max_log_lines} lines.",
        "Prefer discovery before implementation when scope is uncertain.",
    ]
    denied = [
        "Do not scan the entire repository as the first step.",
        "Do not run a full test suite before identifying a focused target.",
        "Do not paste full logs or generated files into the response.",
        "Do not refactor unrelated code while pursuing the task.",
    ]
    stop_loss = [
        f"Stop if more than {max_files} files are needed before a plausible path is found.",
        f"Stop if {max_commands} commands do not reproduce or narrow the issue.",
        "Stop if the likely fix crosses more than 3 modules.",
        "Stop if the token estimate moves to a higher risk tier.",
    ]
    if forecast.task_type == "git_publish":
        denied[1] = "Do not publish files outside the intended repository."
        stop_loss[2] = "Stop if git status contains unrelated changes."
    return BudgetContract(mode, max_files, max_commands, max_log_lines, allowed, denied, stop_loss)


def plan_context_diet(task: str, forecast: Estimate) -> ContextDietPlan:
    task_lower = task.lower()
    if forecast.task_type in {"debugging", "test_repair"}:
        read_first = ["failing test file", "directly referenced module", "test configuration"]
        read_second = ["caller/callee modules", "fixture or mock setup", "recent related changes"]
        avoid = ["unrelated UI pages", "documentation", "build artifacts", "vendor/generated folders"]
    elif forecast.task_type == "git_publish":
        read_first = ["README.md", "SKILL.md", "agents/openai.yaml", "git status"]
        read_second = ["scripts directory", "references directory", "remote configuration"]
        avoid = ["global Codex config", "other local skills", "private history files"]
    elif forecast.task_type in {"feature_development", "frontend_build"}:
        read_first = ["entry point for target feature", "nearest existing component/module", "focused tests or examples"]
        read_second = ["shared helpers", "style/config files", "validation commands"]
        avoid = ["unrelated feature areas", "large snapshots", "full dependency trees"]
    elif forecast.task_type in {"refactor", "large_project_task"}:
        read_first = ["repo map or architecture summary", "target module boundaries", "highest-risk call sites"]
        read_second = ["focused tests", "shared interfaces", "migration examples"]
        avoid = ["full source dumps", "generated files", "unrelated packages"]
    elif "novel" in task_lower or "report" in task_lower or "文章" in task_lower:
        read_first = ["user requirements", "target audience", "outline constraints"]
        read_second = ["style references if supplied", "format requirements"]
        avoid = ["repository files", "tool logs", "unrelated research"]
    else:
        read_first = ["user request", "nearest relevant file or context", "existing examples"]
        read_second = ["supporting configs", "focused verification target"]
        avoid = ["unrelated directories", "full logs", "generated outputs"]
    escalation = "Broaden context only after the first-pass files fail to identify a plausible path."
    return ContextDietPlan(read_first, read_second, avoid, escalation)


def choose_contract_mode(mode: str | None, forecast: Estimate, roi: ROIAssessment) -> str:
    if mode:
        return mode
    if roi.decision in {"discovery_first", "defer"} or forecast.risk in {"high", "extreme"}:
        return "cheap"
    if forecast.risk == "medium":
        return "balanced"
    return "thorough"


def build_first_pass_prompt(task: str, forecast: Estimate, roi: ROIAssessment, contract: BudgetContract, diet: ContextDietPlan) -> str:
    lines = [
        "# Recommended First Pass",
        "",
        f"Task: {compact_task_text(task)}",
        f"Decision: {roi.decision}",
        f"Token budget mode: {contract.mode}",
        "",
        "## Do First",
        *[f"- Read: {item}" for item in diet.read_first],
        "",
        "## Budget Rules",
        *[f"- {item}" for item in contract.allowed_actions],
        "",
        "## Stop-Loss",
        *[f"- {item}" for item in contract.stop_loss],
        "",
        "## Report",
        "- Summarize findings, likely next step, files inspected, commands run, and whether to continue.",
    ]
    if roi.decision == "discovery_first":
        lines.insert(3, "Do not edit files in this pass.")
    return "\n".join(lines)


def build_preflight_controller(
    task: str,
    cwd: str,
    context_tokens: int,
    history_path: Path,
    budget_tokens: int | None = None,
    context_window: int | None = None,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
    currency: str = "USD",
    mode: str | None = None,
) -> PreflightController:
    forecast = estimate(
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
    roi = assess_roi(forecast)
    contract_mode = choose_contract_mode(mode, forecast, roi)
    contract = generate_budget_contract(forecast, contract_mode)
    diet = plan_context_diet(task, forecast)
    first_pass = build_first_pass_prompt(task, forecast, roi, contract, diet)
    return PreflightController(forecast, roi, contract, diet, first_pass)


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


def render_optimization_markdown(result: PromptOptimization) -> str:
    original_low = f"{result.original_estimate.low_tokens // 1000}k" if result.original_estimate.low_tokens >= 1000 else str(result.original_estimate.low_tokens)
    original_high = f"{result.original_estimate.high_tokens // 1000}k" if result.original_estimate.high_tokens >= 1000 else str(result.original_estimate.high_tokens)
    optimized_low = f"{result.optimized_estimate.low_tokens // 1000}k" if result.optimized_estimate.low_tokens >= 1000 else str(result.optimized_estimate.low_tokens)
    optimized_high = f"{result.optimized_estimate.high_tokens // 1000}k" if result.optimized_estimate.high_tokens >= 1000 else str(result.optimized_estimate.high_tokens)
    lines = [
        "**Budget-Aware Prompt Optimization**",
        f"- Mode: {result.mode}",
        f"- Original estimate: {original_low}-{original_high} tokens",
        f"- Optimized estimate: {optimized_low}-{optimized_high} tokens",
        f"- Original risk: {result.original_estimate.risk}",
        f"- Optimized risk: {result.optimized_estimate.risk}",
        f"- Savings: {result.savings_summary}",
        "",
        "## Optimized Prompt",
        "",
        result.optimized_prompt,
    ]
    return "\n".join(lines)


def render_controller_markdown(result: PreflightController) -> str:
    forecast_low = f"{result.forecast.low_tokens // 1000}k" if result.forecast.low_tokens >= 1000 else str(result.forecast.low_tokens)
    forecast_high = f"{result.forecast.high_tokens // 1000}k" if result.forecast.high_tokens >= 1000 else str(result.forecast.high_tokens)
    lines = [
        "**Codex Token Preflight Controller**",
        "",
        "## Token Forecast",
        f"- Estimate: {forecast_low}-{forecast_high} tokens",
        f"- Risk: {result.forecast.risk}",
        f"- Task type: {result.forecast.task_type}",
        f"- Main drivers: {', '.join(result.forecast.drivers)}",
        "",
        "## ROI Assessor",
        f"- ROI: {result.roi.label} ({result.roi.score}/100)",
        f"- Decision: {result.roi.decision}",
        f"- Expected value: {result.roi.expected_value}",
        f"- Cost pressure: {result.roi.cost_pressure}",
        *[f"- {item}" for item in result.roi.rationale],
        "",
        "## Budget Contract",
        f"- Mode: {result.contract.mode}",
        f"- Max files before first report: {result.contract.max_files}",
        f"- Max commands before reassessment: {result.contract.max_commands}",
        f"- Max log lines: {result.contract.max_log_lines}",
        "- Allowed:",
        *[f"  - {item}" for item in result.contract.allowed_actions],
        "- Denied:",
        *[f"  - {item}" for item in result.contract.denied_actions],
        "- Stop-loss:",
        *[f"  - {item}" for item in result.contract.stop_loss],
        "",
        "## Context Diet Plan",
        "- Read first:",
        *[f"  - {item}" for item in result.diet.read_first],
        "- Read second:",
        *[f"  - {item}" for item in result.diet.read_second],
        "- Avoid initially:",
        *[f"  - {item}" for item in result.diet.avoid_initially],
        f"- Escalation: {result.diet.escalation_rule}",
        "",
        "## Recommended First Pass Prompt",
        "",
        result.recommended_first_pass_prompt,
    ]
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

    optimize_parser = sub.add_parser("optimize")
    optimize_parser.add_argument("--task", required=True)
    optimize_parser.add_argument("--cwd", default=os.getcwd())
    optimize_parser.add_argument("--mode", choices=["cheap", "balanced", "thorough"], default=None)
    optimize_parser.add_argument("--context-tokens", type=int, default=None)
    optimize_parser.add_argument("--budget", type=int, default=None)
    optimize_parser.add_argument("--context-window", type=int, default=None)
    optimize_parser.add_argument("--input-price-per-million", type=float, default=None)
    optimize_parser.add_argument("--output-price-per-million", type=float, default=None)
    optimize_parser.add_argument("--currency", default=None)
    optimize_parser.add_argument("--history", type=Path, default=default_history_path())
    optimize_parser.add_argument("--json", action="store_true")

    control_parser = sub.add_parser("control")
    control_parser.add_argument("--task", required=True)
    control_parser.add_argument("--cwd", default=os.getcwd())
    control_parser.add_argument("--mode", choices=["cheap", "balanced", "thorough"], default=None)
    control_parser.add_argument("--context-tokens", type=int, default=None)
    control_parser.add_argument("--budget", type=int, default=None)
    control_parser.add_argument("--context-window", type=int, default=None)
    control_parser.add_argument("--input-price-per-million", type=float, default=None)
    control_parser.add_argument("--output-price-per-million", type=float, default=None)
    control_parser.add_argument("--currency", default=None)
    control_parser.add_argument("--history", type=Path, default=default_history_path())
    control_parser.add_argument("--json", action="store_true")

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

    if args.command == "optimize":
        config = load_project_config(args.cwd)
        result = optimize_task(
            args.task,
            args.cwd,
            int(option_value(args.context_tokens, config.context_tokens, 0)),
            args.history,
            option_value(args.budget, config.budget_tokens),
            option_value(args.context_window, config.context_window),
            option_value(args.input_price_per_million, config.input_price_per_million),
            option_value(args.output_price_per_million, config.output_price_per_million),
            str(option_value(args.currency, config.currency, "USD")),
            args.mode,
        )
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print(render_optimization_markdown(result))
        return 0

    if args.command == "control":
        config = load_project_config(args.cwd)
        result = build_preflight_controller(
            args.task,
            args.cwd,
            int(option_value(args.context_tokens, config.context_tokens, 0)),
            args.history,
            option_value(args.budget, config.budget_tokens),
            option_value(args.context_window, config.context_window),
            option_value(args.input_price_per_million, config.input_price_per_million),
            option_value(args.output_price_per_million, config.output_price_per_million),
            str(option_value(args.currency, config.currency, "USD")),
            args.mode,
        )
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print(render_controller_markdown(result))
        return 0

    if args.command == "record":
        record(args.history, args)
        print(f"Recorded token budget sample to {args.history}")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
