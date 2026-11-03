#!/usr/bin/env python3
"""Estimate full agent-task token usage before execution."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import history_model
import preflight_model

TASK_PROFILES = {
    "code_repair": {"complexity": 4, "searches": (5, 14), "files": (6, 20), "commands": (2, 6), "failures": (1, 4)},
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
    model: str | None = None
    agent: str | None = None
    execution_config: str | None = None
    peak_context_tokens: int | None = None
    task_kind: str = "auto"
    predictor: str = "adaptive"
    point_model: str = "auto"


@dataclass
class Estimate:
    task_type: str
    complexity: int
    risk: str
    confidence: str
    low_tokens: int
    high_tokens: int
    midpoint_tokens: int
    input_low_tokens: int | None
    input_high_tokens: int | None
    output_low_tokens: int | None
    output_high_tokens: int | None
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

    task_text: str = ""
    baseline_version: str = history_model.BASELINE_VERSION
    baseline_midpoint_tokens: int = 0
    prediction_method: str = "heuristic"
    p50_tokens: int | None = None
    p90_tokens: int | None = None
    interval_method: str = "heuristic_uncalibrated"
    interval_target_coverage: float | None = None
    fit_samples: int = 0
    calibration_samples: int = 0
    history_digest: str | None = None
    execution_profile: dict[str, str] | None = None
    instance_id: str | None = None
    peak_context_tokens: int | None = None
    evidence_note: str = "Rule-based rough estimate; no calibrated coverage."

    feature_snapshot: dict | None = None
    point_method: str = "heuristic"
    point_semantics: str = "uncalibrated_median_proxy"
    budget_upper_tokens: int | None = None
    budget_target_coverage: float | None = None
    model_selection: str | None = None


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
            model=data.get("model"), agent=data.get("agent"),
            execution_config=data.get("execution_config"),
            peak_context_tokens=_optional_int(data.get("peak_context_tokens")),
            task_kind=str(data.get("task_kind", "auto")), predictor=str(data.get("predictor", "adaptive")),
            point_model=str(data.get("point_model", "auto")),
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
            lowered = str(path.relative_to(root)).lower()
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


def contains_term(text: str, term: str) -> bool:
    if term.isascii():
        return re.search(r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])", text) is not None
    return term in text


def load_repo_signals(path: Path) -> RepoSignals:
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = ["file_count", "code_file_count", "test_file_count", "config_file_count"]
    if not isinstance(data, dict) or not all(isinstance(data.get(k), int) and not isinstance(data[k], bool)
                                           and data[k] >= 0 for k in fields):
        raise ValueError("repository counts must be nonnegative integers")
    if any(data[k] > data["file_count"] for k in fields[1:]):
        raise ValueError("component counts cannot exceed total file count")
    count = data["code_file_count"]
    size = "small" if count < 50 else "medium" if count < 500 else "large" if count < 2000 else "very_large"
    return RepoSignals(*(data[k] for k in fields), repo_size=size)


def classify_task(task: str, task_kind: str = "auto") -> str:
    if task_kind == "code_repair":
        return "code_repair"
    if task_kind != "auto":
        raise ValueError("task kind must be auto or code_repair")
    t = task.lower()
    has_git_publish = any(contains_term(t, w) for w in [
        "upload to github", "publish to github", "push to github", "create github repo",
        "github repo", "git remote", "git push", "open source", "release repo",
        "publish skill", "upload skill", "push the branch", "create or configure remote",
    ])
    has_code = any(contains_term(t, w) for w in ["code", "function", "class", "repo", "project", "代码", "函数", "项目"])
    has_test = any(contains_term(t, w) for w in ["test", "pytest", "jest", "ci", "failing", "failure", "测试", "报错"])
    has_bug = any(contains_term(t, w) for w in ["bug", "debug", "fix", "error", "exception", "traceback", "修复", "错误", "失败"])
    has_frontend = any(contains_term(t, w) for w in ["frontend", "ui", "react", "vue", "page", "dashboard", "browser", "前端", "页面"])
    has_refactor = any(contains_term(t, w) for w in ["refactor", "rewrite", "restructure", "重构", "改造"])
    has_research = any(contains_term(t, w) for w in ["research", "search", "look up", "调研", "搜索", "查一下"])
    has_feature = any(contains_term(t, w) for w in ["add", "build", "create", "implement", "feature", "新增", "实现", "开发", "生成"])
    broad = any(contains_term(t, w) for w in ["entire", "all", "full", "complete", "全量", "整个", "全部", "完整"])

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


def context_fit_for(high_tokens: int | None, context_window: int | None) -> str | None:
    if context_window is None:
        return None
    if high_tokens is None:
        return "unknown"
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
    execution_profile: dict[str, str] | None = None,
    peak_context_tokens: int | None = None,
    instance_id: str | None = None,
    task_kind: str = "auto",
    predictor: str = "adaptive",
    point_model: str = "auto",
    repo_signals: RepoSignals | None = None,
) -> Estimate:
    if context_tokens < 0:
        raise ValueError("initial context must be nonnegative")
    repo = repo_signals if repo_signals is not None else collect_repo_signals(cwd)
    task_type = classify_task(task, task_kind)
    profile = dict(TASK_PROFILES[task_type])
    complexity = int(profile["complexity"])

    if repo.repo_size == "large":
        complexity += 1
    elif repo.repo_size == "very_large":
        complexity += 2
    if repo.test_file_count > 0 and task_type in {"debugging", "test_repair", "feature_development", "refactor"}:
        complexity += 1
    if any(contains_term(task.lower(), w) for w in ["all", "entire", "full", "complete", "全部", "整个", "完整"]):
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

    # Versioned raw scale; never calibrate against a previously adjusted forecast.
    low_input = int(low_input * 0.85)
    low_output = int(low_output * 0.85)
    high_input = int(high_input * 1.25)
    high_output = int(high_output * 1.25)
    low, high = low_input + low_output, high_input + high_output
    baseline_midpoint = round_to_nice(int((low + high) / 2))
    profile = execution_profile or {}
    features = preflight_model.feature_snapshot(task, repo.file_count, repo.code_file_count,
                                                repo.test_file_count, repo.repo_size != "unknown", context_tokens)
    historical = history_model.predict(history_path, task_type, profile, baseline_midpoint,
                                       instance_id=instance_id, task=task, features=features,
                                       strategy=predictor, point_model=point_model)
    ratio = historical.p50_tokens / baseline_midpoint if historical.p50_tokens is not None else 1.0
    if historical.calibrated:
        low, high = historical.low_tokens, historical.high_tokens
    elif historical.p50_tokens is not None:
        if getattr(historical, "interval_method", "").startswith("one_sided"):
            low, high = 0, historical.high_tokens
        else:
            low, high = int(low * ratio), max(int(high * ratio), historical.p90_tokens or 0)
    initial_tokens = prompt_low + context_tokens
    # Estimated prompt length is not a known lower bound on actual usage.
    # Do not shrink a calibrated interval using that approximation.
    if not historical.calibrated and not getattr(historical, "interval_method", "").startswith("one_sided"):
        low = max(initial_tokens, low)
    high = max(initial_tokens, high)
    midpoint = int((low + high) / 2)
    confidence = "unvalidated" if historical.p50_tokens is not None else "low"
    rounded_high = round_to_nice(high)
    risk = risk_for(rounded_high)
    budget_status = budget_status_for(rounded_high, budget_tokens)
    rounded_input_low = round_to_nice(low_input)
    rounded_input_high = round_to_nice(high_input)
    rounded_output_low = round_to_nice(low_output)
    rounded_output_high = round_to_nice(high_output)
    context_fit = context_fit_for(peak_context_tokens, context_window)
    cost = cost_for(
        rounded_input_low,
        rounded_input_high,
        rounded_output_low,
        rounded_output_high,
        input_price_per_million,
        output_price_per_million,
        currency,
    )

    if historical.p50_tokens is not None:
        rounded_input_low = rounded_input_high = None
        rounded_output_low = rounded_output_high = None
        cost = None

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
        low_tokens=int(low) if historical.calibrated else round_to_nice(low),
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
        task_text=task, baseline_midpoint_tokens=baseline_midpoint,
        prediction_method=historical.method,
        p50_tokens=historical.p50_tokens, p90_tokens=historical.p90_tokens,
        interval_method=getattr(historical, "interval_method", "split_conformal_90" if historical.calibrated else "heuristic_uncalibrated"),
        interval_target_coverage=historical.target_coverage,
        fit_samples=historical.fit_samples, calibration_samples=historical.calibration_samples,
        history_digest=historical.history_digest, execution_profile=profile,
        instance_id=instance_id, peak_context_tokens=peak_context_tokens,
        feature_snapshot=features, point_method=getattr(historical, "point_method", historical.method),
        budget_upper_tokens=rounded_high if getattr(historical, "budget_upper_tokens", None) is not None else None,
        budget_target_coverage=historical.target_coverage if getattr(historical, "budget_upper_tokens", None) is not None else None,
        model_selection=getattr(historical, "selection", None),
        evidence_note=("Target 90% marginal coverage assumes exchangeable, same-configuration tasks. One-sided budget coverage is not central interval coverage; point accuracy is unvalidated."
                       if historical.calibrated else
                       "Uncalibrated scenario bound. The point is a median proxy, not a validated quantile or success probability."
                       if historical.p50_tokens is not None else
                       "Insufficient matching fit history (need 5 independent tasks and a complete execution profile); using rough rules."),
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
    execution_profile: dict[str, str] | None = None,
    peak_context_tokens: int | None = None,
    instance_id: str | None = None,
    task_kind: str = "auto",
    predictor: str = "adaptive",
    point_model: str = "auto",
    repo_signals: RepoSignals | None = None,
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
            execution_profile=execution_profile, peak_context_tokens=peak_context_tokens, instance_id=instance_id,
            task_kind=task_kind, predictor=predictor, point_model=point_model, repo_signals=repo_signals,
        )
        for task in tasks
    ]
    results.sort(key=lambda item: item.p50_tokens if item.p50_tokens is not None else item.midpoint_tokens, reverse=True)
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
        },
        "balanced": {
            "label": "balanced",
            "goal": "Complete the requested task with scoped discovery, focused edits, and focused validation.",
            "verification": "Run focused validation first, then one broader check if the focused check passes.",
            "max_files": "Inspect only files connected to the target behavior before broadening scope.",
            "max_commands": "Run focused commands before broad test or build commands.",
            "output_cap": "Summarize command output and include only actionable failure excerpts.",
        },
        "thorough": {
            "label": "thorough",
            "goal": "Complete the task carefully while still avoiding unrelated repository exploration.",
            "verification": "Run focused validation plus broader regression checks where available.",
            "max_files": "Inspect related modules, tests, and configs before implementation.",
            "max_commands": "Run necessary test/build commands, but cap repeated failures.",
            "output_cap": "Summarize logs by failure chain and avoid dumping full repeated output.",
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
    execution_profile: dict[str, str] | None = None,
    peak_context_tokens: int | None = None,
    instance_id: str | None = None,
    task_kind: str = "auto",
    predictor: str = "adaptive",
    point_model: str = "auto",
    repo_signals: RepoSignals | None = None,
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
        execution_profile=execution_profile, peak_context_tokens=peak_context_tokens, instance_id=instance_id,
            task_kind=task_kind, predictor=predictor, point_model=point_model, repo_signals=repo_signals,
    )
    chosen_mode = choose_mode(mode, original)
    optimized_prompt, auto_split, scope_guard, output_caps = build_optimized_prompt(task, original, chosen_mode)
    optimized = original
    savings_summary = "Savings not measured; the suggested prompt does not change the forecast until independently validated."
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
    context_penalty = {"fits": 0, "tight": 12, "likely_exceeds": 25, "unknown": 0, None: 0}[forecast.context_fit]
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
    execution_profile: dict[str, str] | None = None,
    peak_context_tokens: int | None = None,
    instance_id: str | None = None,
    task_kind: str = "auto",
    predictor: str = "adaptive",
    point_model: str = "auto",
    repo_signals: RepoSignals | None = None,
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
        execution_profile=execution_profile, peak_context_tokens=peak_context_tokens, instance_id=instance_id,
            task_kind=task_kind, predictor=predictor, point_model=point_model, repo_signals=repo_signals,
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
        f"- Method: {result.prediction_method}; interval: {result.interval_method}",
        f"- Matching tasks: {result.fit_samples} fit / {result.calibration_samples} calibration",
        f"- Point model: {result.point_method}; selection: {result.model_selection}",
        f"- Evidence: {result.evidence_note}",
        f"- Task type: {result.task_type}",
        f"- Complexity: {result.complexity}/10",
        f"- Input tokens: {result.input_low_tokens}-{result.input_high_tokens}" if result.input_low_tokens is not None else "- Input tokens: unknown after total-only historical calibration",
        f"- Output tokens: {result.output_low_tokens}-{result.output_high_tokens}" if result.output_low_tokens is not None else "- Output tokens: unknown after total-only historical calibration",
        f"- Repo signals: {result.repo.repo_size}, {result.repo.code_file_count} code files, {result.repo.test_file_count} test-like files",
        f"- Main drivers: {', '.join(result.drivers)}",
        f"- Recommendation: {result.recommendation}",
    ]
    if result.p50_tokens is not None:
        lines.append(f"- Median proxy: {result.p50_tokens} tokens; historical P90: {result.p90_tokens}")
    if result.budget_upper_tokens is not None:
        lines.append(f"- One-sided budget upper: {result.budget_upper_tokens} tokens (90% marginal target under assumptions)")
    if result.context_window is not None:
        lines.append(f"- Peak context fit: {result.context_fit} against {result.context_window} tokens (separate from cumulative usage)")
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
    lines = ["**Token Budget Comparison**", "", "| Rank | Estimate | P50 | Interval method | Risk | Type | Recommendation |", "|---:|---|---|---|---|---|---|"]
    for result in results:
        low = f"{result.low_tokens // 1000}k" if result.low_tokens >= 1000 else str(result.low_tokens)
        high = f"{result.high_tokens // 1000}k" if result.high_tokens >= 1000 else str(result.high_tokens)
        lines.append(
            f"| {result.rank} | {low}-{high} | {result.p50_tokens if result.p50_tokens is not None else 'unknown'} | {result.interval_method} | {result.risk} | {result.task_type} | {result.recommendation} |"
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
        f"- Forecast retained pending measurement: {optimized_low}-{optimized_high} tokens",
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
        f"- Method: {result.forecast.prediction_method}; interval: {result.forecast.interval_method}",
        f"- Evidence: {result.forecast.evidence_note}",
        f"- Median proxy: {result.forecast.p50_tokens}; one-sided budget upper: {result.forecast.budget_upper_tokens}",
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
    if not history_model.positive_number(args.actual_tokens):
        raise ValueError("actual tokens must be a positive finite integer")
    if args.prediction_file is not None:
        prediction = json.loads(args.prediction_file.read_text(encoding="utf-8"))
        if not isinstance(prediction, dict) or prediction.get("baseline_version") != history_model.BASELINE_VERSION:
            raise ValueError("prediction file must be a current-version estimate JSON")
        profile = prediction.get("execution_profile")
        if not isinstance(profile, dict) or not all(isinstance(profile.get(k), str) and profile[k].strip()
                                                   for k in history_model.PROFILE_KEYS):
            raise ValueError("prediction file needs model, agent and execution-config")
        for key in history_model.PROFILE_KEYS:
            value = getattr(args, key)
            if value is not None and value != profile[key]:
                raise ValueError("record configuration must match the frozen prediction")
        identity = args.instance_id or prediction.get("instance_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("record requires a stable --instance-id")
        if prediction.get("instance_id") and identity != prediction["instance_id"]:
            raise ValueError("instance ID must match the frozen prediction")
        baseline = prediction.get("baseline_midpoint_tokens")
        if not history_model.positive_number(baseline):
            raise ValueError("prediction file needs a positive raw baseline")
        task, task_type = prediction.get("task_text"), prediction.get("task_type")
        if not isinstance(task, str) or not task.strip() or task_type not in TASK_PROFILES:
            raise ValueError("prediction file needs task_text and a known task_type")
        if args.task is not None and args.task != task:
            raise ValueError("record task must match the frozen prediction")
        if args.task_type is not None and args.task_type != task_type:
            raise ValueError("record task type must match the frozen prediction")
        row = {
            "schema_version": 2, "baseline_version": history_model.BASELINE_VERSION,
            "instance_id": identity, "task": task, "task_type": task_type,
            "execution_profile": profile, "baseline_midpoint_tokens": baseline,
            "actual_tokens": args.actual_tokens, "split": args.split, "outcome": args.outcome,
            "prediction": prediction, "feature_snapshot": prediction.get("feature_snapshot"),
        }
    else:
        required = [args.task, args.task_type, args.predicted_low, args.predicted_high, args.predicted_midpoint]
        if any(v is None for v in required):
            raise ValueError("use --prediction-file, or supply all legacy task/predicted flags")
        if not all(history_model.positive_number(v) for v in required[2:]):
            raise ValueError("legacy predicted token values must be positive")
        if not args.predicted_low <= args.predicted_midpoint <= args.predicted_high:
            raise ValueError("legacy prediction must satisfy low <= midpoint <= high")
        row = {
            "schema_version": 1, "task": args.task, "task_type": args.task_type,
            "predicted_low_tokens": args.predicted_low, "predicted_high_tokens": args.predicted_high,
            "predicted_midpoint_tokens": args.predicted_midpoint, "actual_tokens": args.actual_tokens,
        }
    row["recorded_at"] = datetime.now(timezone.utc).isoformat()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


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
    record_parser.add_argument("--task")
    record_parser.add_argument("--task-type")
    record_parser.add_argument("--predicted-low", type=int)
    record_parser.add_argument("--predicted-high", type=int)
    record_parser.add_argument("--predicted-midpoint", type=int)
    record_parser.add_argument("--actual-tokens", type=int, required=True)
    record_parser.add_argument("--history", type=Path, default=default_history_path())

    for command_parser in [estimate_parser, compare_parser, optimize_parser, control_parser, record_parser]:
        command_parser.add_argument("--model", default=None, help="Exact model/version for history matching.")
        command_parser.add_argument("--agent", default=None, help="Execution harness and version.")
        command_parser.add_argument("--execution-config", default=None, help="Versioned reasoning/context/tool configuration ID.")
        command_parser.add_argument("--instance-id", default=None, help="Stable task ID; excludes this task from history fitting.")
    for command_parser in [estimate_parser, compare_parser, optimize_parser, control_parser]:
        command_parser.add_argument("--task-kind", choices=["auto", "code_repair"], default=None)
        command_parser.add_argument("--predictor", choices=["adaptive", "legacy"], default=None)
        command_parser.add_argument("--point-model", choices=["auto", "median", "ridge"], default=None)
        command_parser.add_argument("--repo-signals-file", type=Path, default=None,
                                    help="Declared counts from the exact repository snapshot; no labels.")
        command_parser.add_argument("--peak-context-tokens", type=int, default=None,
                                    help="Separately supplied peak single-request context estimate.")
    record_parser.add_argument("--prediction-file", type=Path, help="Frozen JSON from estimate; recommended for v2 history.")
    record_parser.add_argument("--split", choices=["auto", "fit", "calibration", "test"], default="auto")
    record_parser.add_argument("--outcome", choices=["completed", "failed", "capped", "environment_error"], default="completed")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "estimate":
        config = load_project_config(args.cwd)
        profile = {"model": option_value(args.model, config.model),
                   "agent": option_value(args.agent, config.agent),
                   "execution_config": option_value(args.execution_config, config.execution_config)}
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
            execution_profile=profile, peak_context_tokens=option_value(args.peak_context_tokens, config.peak_context_tokens),
            instance_id=args.instance_id,
            task_kind=option_value(args.task_kind, config.task_kind),
            predictor=option_value(args.predictor, config.predictor),
            point_model=option_value(args.point_model, config.point_model),
            repo_signals=load_repo_signals(args.repo_signals_file) if args.repo_signals_file else None,
        )
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print(render_markdown(result))
        return 0

    if args.command == "compare":
        config = load_project_config(args.cwd)
        profile = {"model": option_value(args.model, config.model),
                   "agent": option_value(args.agent, config.agent),
                   "execution_config": option_value(args.execution_config, config.execution_config)}
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
            execution_profile=profile, peak_context_tokens=option_value(args.peak_context_tokens, config.peak_context_tokens),
            instance_id=args.instance_id,
            task_kind=option_value(args.task_kind, config.task_kind),
            predictor=option_value(args.predictor, config.predictor),
            point_model=option_value(args.point_model, config.point_model),
            repo_signals=load_repo_signals(args.repo_signals_file) if args.repo_signals_file else None,
        )
        if args.json:
            print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
        else:
            print(render_compare_markdown(results))
        return 0

    if args.command == "optimize":
        config = load_project_config(args.cwd)
        profile = {"model": option_value(args.model, config.model),
                   "agent": option_value(args.agent, config.agent),
                   "execution_config": option_value(args.execution_config, config.execution_config)}
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
            execution_profile=profile, peak_context_tokens=option_value(args.peak_context_tokens, config.peak_context_tokens),
            instance_id=args.instance_id,
            task_kind=option_value(args.task_kind, config.task_kind),
            predictor=option_value(args.predictor, config.predictor),
            point_model=option_value(args.point_model, config.point_model),
            repo_signals=load_repo_signals(args.repo_signals_file) if args.repo_signals_file else None,
        )
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print(render_optimization_markdown(result))
        return 0

    if args.command == "control":
        config = load_project_config(args.cwd)
        profile = {"model": option_value(args.model, config.model),
                   "agent": option_value(args.agent, config.agent),
                   "execution_config": option_value(args.execution_config, config.execution_config)}
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
            execution_profile=profile, peak_context_tokens=option_value(args.peak_context_tokens, config.peak_context_tokens),
            instance_id=args.instance_id,
            task_kind=option_value(args.task_kind, config.task_kind),
            predictor=option_value(args.predictor, config.predictor),
            point_model=option_value(args.point_model, config.point_model),
            repo_signals=load_repo_signals(args.repo_signals_file) if args.repo_signals_file else None,
        )
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print(render_controller_markdown(result))
        return 0

    if args.command == "record":
        try:
            record(args.history, args)
        except (ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        print(f"Recorded token budget sample to {args.history}")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
