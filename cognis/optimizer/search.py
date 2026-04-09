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


SEARCH_TRACE_SCHEMA_VERSION = "objective_search_trace_v1"


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
    candidate_count: int
    best_index: int
    best_params: dict[str, float]
    best_score: float
    score_margin_to_next: float | None
    evaluations: tuple[SearchCandidateEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        return _to_builtin(asdict(self))


def _grid_candidates():
    brightness_grid = [-0.2, 0.0, 0.2]
    width_grid = [0.9, 1.0, 1.1]
    bass_preservation_grid = [0.8, 1.0]
    dynamics_preservation_grid = [0.8, 1.0]

    for brightness in brightness_grid:
        for width in width_grid:
            for bass_preservation in bass_preservation_grid:
                for dynamics_preservation in dynamics_preservation_grid:
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
    trace = SearchTrace(
        schema_version=SEARCH_TRACE_SCHEMA_VERSION,
        selection_basis="exact_bounded_grid_search",
        candidate_count=len(evaluations),
        best_index=best_index,
        best_params=best_params,
        best_score=best_score,
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
