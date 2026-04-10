from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict

import numpy as np

from cognis.analysis.analyzer import Analyzer
from cognis.optimizer.objective import (
    ObjectiveAttribution,
    build_objective_attribution,
    compute_objective,
)
from cognis.optimizer.targets import TargetValues


SEARCH_TRACE_SCHEMA_VERSION = "objective_search_trace_v2"

GRID_AXES: dict[str, tuple[float, ...]] = {
    "brightness": (-0.2, 0.0, 0.2),
    "width": (0.9, 1.0, 1.1),
    "bass_preservation": (0.8, 1.0),
    "dynamics_preservation": (0.8, 1.0),
}


def _to_builtin(value):
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class SearchCandidateEvaluation:
    index: int
    params: dict[str, float]
    score: float
    attribution: ObjectiveAttribution

    def to_dict(self) -> dict[str, object]:
        return _to_builtin(asdict(self))


@dataclass(frozen=True)
class SearchTrace:
    schema_version: str
    selection_basis: str
    parameter_axes: dict[str, tuple[float, ...]]
    candidate_count: int
    ranking_rule: str
    ranked_candidate_indexes: tuple[int, ...]
    best_index: int
    best_params: dict[str, float]
    best_score: float
    runner_up_index: int | None
    runner_up_score: float | None
    winner_score_margin_to_runner_up: float | None
    tie_count_at_best_score: int
    score_margin_to_next: float | None
    evaluations: tuple[SearchCandidateEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        return _to_builtin(asdict(self))


def _grid_candidates():
    for brightness in GRID_AXES["brightness"]:
        for width in GRID_AXES["width"]:
            for bass_preservation in GRID_AXES["bass_preservation"]:
                for dynamics_preservation in GRID_AXES["dynamics_preservation"]:
                    yield {
                        "brightness": brightness,
                        "width": width,
                        "bass_preservation": bass_preservation,
                        "dynamics_preservation": dynamics_preservation,
                    }


def _best_margin(scores: list[float]) -> float | None:
    if len(scores) < 2:
        return None
    ordered = sorted(scores)
    return float(ordered[1] - ordered[0])


def _ranked_indexes(evaluations: list[SearchCandidateEvaluation]) -> tuple[int, ...]:
    return tuple(evaluation.index for evaluation in sorted(evaluations, key=lambda item: (item.score, item.index)))


def _runner_up(evaluations: list[SearchCandidateEvaluation]) -> SearchCandidateEvaluation | None:
    ranked = sorted(evaluations, key=lambda item: (item.score, item.index))
    if len(ranked) < 2:
        return None
    return ranked[1]


def _tie_count_at_best_score(evaluations: list[SearchCandidateEvaluation], best_score: float) -> int:
    return sum(1 for evaluation in evaluations if abs(evaluation.score - best_score) <= 1e-12)


def _grid_search_impl(
    audio: np.ndarray,
    sr: int,
    targets: TargetValues,
    render_fn: Callable[[np.ndarray, Dict[str, float]], np.ndarray],
    analyzer: Analyzer,
    *,
    with_trace: bool,
) -> tuple[dict[str, float], SearchTrace | None]:
    best_score = float("inf")
    best_params = {"brightness": 0.0, "width": 1.0, "bass_preservation": 1.0, "dynamics_preservation": 1.0}
    best_index = -1
    evaluations: list[SearchCandidateEvaluation] = []

    for index, params in enumerate(_grid_candidates()):
        rendered = render_fn(audio, params)
        analysis = analyzer.analyze(rendered, sr)
        score = compute_objective(analysis, targets)

        if with_trace:
            evaluations.append(
                SearchCandidateEvaluation(
                    index=index,
                    params=dict(sorted(params.items())),
                    score=score,
                    attribution=build_objective_attribution(analysis, targets),
                )
            )

        if score < best_score:
            best_score = score
            best_params = dict(sorted(params.items()))
            best_index = index

    if not with_trace:
        return best_params, None

    trace_scores = [evaluation.score for evaluation in evaluations]
    runner_up = _runner_up(evaluations)
    trace = SearchTrace(
        schema_version=SEARCH_TRACE_SCHEMA_VERSION,
        selection_basis="exact_bounded_grid_search",
        parameter_axes=GRID_AXES,
        candidate_count=len(evaluations),
        ranking_rule="sort_by_score_then_index",
        ranked_candidate_indexes=_ranked_indexes(evaluations),
        best_index=best_index,
        best_params=best_params,
        best_score=best_score,
        runner_up_index=None if runner_up is None else runner_up.index,
        runner_up_score=None if runner_up is None else runner_up.score,
        winner_score_margin_to_runner_up=(
            None if runner_up is None else float(runner_up.score - best_score)
        ),
        tie_count_at_best_score=_tie_count_at_best_score(evaluations, best_score),
        score_margin_to_next=_best_margin(trace_scores),
        evaluations=tuple(evaluations),
    )
    return best_params, trace


def grid_search(
    audio: np.ndarray,
    sr: int,
    targets: TargetValues,
    render_fn: Callable[[np.ndarray, Dict[str, float]], np.ndarray],
    analyzer: Analyzer,
) -> Dict[str, float]:
    """
    Bounded first-pass search over a small space.
    render_fn takes (audio, params) and returns rendered audio.
    """

    best_params, _ = _grid_search_impl(
        audio,
        sr,
        targets,
        render_fn,
        analyzer,
        with_trace=False,
    )
    return best_params


def grid_search_with_trace(
    audio: np.ndarray,
    sr: int,
    targets: TargetValues,
    render_fn: Callable[[np.ndarray, Dict[str, float]], np.ndarray],
    analyzer: Analyzer,
) -> SearchTrace:
    best_params, trace = _grid_search_impl(
        audio,
        sr,
        targets,
        render_fn,
        analyzer,
        with_trace=True,
    )
    if trace is None:
        raise RuntimeError("Search trace was not built")
    return trace
