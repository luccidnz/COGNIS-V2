from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognis.optimizer.search import SearchCandidateEvaluation, SearchTrace
from cognis.optimizer.targets import TargetValues


DECISION_HISTORY_SCHEMA_VERSION = "decision_history_schema_v1"

REFERENCE_METRIC_TERM_NAMES: dict[str, str] = {
    "integrated_lufs": "reference_integrated_lufs",
    "spectral_tilt_db_per_decade": "reference_spectral_tilt_db_per_decade",
    "low_band_width": "reference_low_band_width",
    "crest_factor_db": "reference_crest_factor_db",
}


def _to_builtin(value: Any) -> Any:
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
class DecisionHistoryLimitation:
    code: str
    message: str
    evidence_level: str


@dataclass(frozen=True)
class DecisionHistorySearch:
    search_trace_schema_version: str
    objective_attribution_schema_version: str
    selection_basis: str
    space_kind: str
    exhaustive_within_space: bool
    global_optimum_claim: str
    parameter_axes: dict[str, tuple[float, ...]]
    candidate_count: int
    ranking_rule: str
    best_tie_break_rule: str


@dataclass(frozen=True)
class DecisionHistoryContext:
    reference_available: bool
    reference_targeting_schema_version: str | None


@dataclass(frozen=True)
class DecisionHistorySelection:
    winner_candidate_index: int
    runner_up_candidate_index: int | None
    winner_score: float
    runner_up_score: float | None
    score_margin_to_runner_up: float | None
    tie_count_at_best_score: int


@dataclass(frozen=True)
class DecisionHistoryCandidateSummary:
    index: int
    rank: int
    params: dict[str, float]
    score: float
    objective_attribution: dict[str, Any]


@dataclass(frozen=True)
class DecisionHistorySeparationTerm:
    term: str
    category: str
    winner_penalty: float
    alternative_penalty: float
    penalty_delta: float
    evidence_level: str


@dataclass(frozen=True)
class DecisionHistoryTradeoff:
    comparison_basis: str
    candidate_index: int
    score_delta_vs_winner: float
    separation_terms: tuple[DecisionHistorySeparationTerm, ...]
    summary: str
    summary_level: str


@dataclass(frozen=True)
class DecisionHistoryMetricTradeoff:
    metric: str
    status: str
    evidence_level: str
    winner_delta_to_reference: float | None
    alternative_candidate_index: int | None
    alternative_delta_to_reference: float | None
    alternative_score_delta_vs_winner: float | None
    blocking_terms: tuple[DecisionHistorySeparationTerm, ...]
    summary: str
    summary_level: str


@dataclass(frozen=True)
class DecisionHistorySummary:
    schema_version: str
    available: bool
    selection_basis: str | None
    candidate_count: int | None
    winner_candidate_index: int | None
    runner_up_candidate_index: int | None
    score_margin_to_runner_up: float | None
    dominant_tradeoffs: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class DecisionHistoryArtifact:
    schema_version: str
    artifact_type: str
    search: DecisionHistorySearch
    context: DecisionHistoryContext
    selection: DecisionHistorySelection
    winner: DecisionHistoryCandidateSummary
    runner_up: DecisionHistoryCandidateSummary | None
    competitive_alternatives: tuple[DecisionHistoryCandidateSummary, ...]
    selection_tradeoffs: tuple[DecisionHistoryTradeoff, ...]
    reference_metric_tradeoffs: tuple[DecisionHistoryMetricTradeoff, ...]
    limitations: tuple[DecisionHistoryLimitation, ...]
    evaluated_candidates: tuple[DecisionHistoryCandidateSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))

    def to_summary(self) -> DecisionHistorySummary:
        dominant_tradeoffs = tuple(tradeoff.summary for tradeoff in self.selection_tradeoffs[:2])
        if not dominant_tradeoffs:
            dominant_tradeoffs = tuple(item.summary for item in self.reference_metric_tradeoffs[:2])
        return DecisionHistorySummary(
            schema_version=self.schema_version,
            available=True,
            selection_basis=self.search.selection_basis,
            candidate_count=self.search.candidate_count,
            winner_candidate_index=self.selection.winner_candidate_index,
            runner_up_candidate_index=self.selection.runner_up_candidate_index,
            score_margin_to_runner_up=self.selection.score_margin_to_runner_up,
            dominant_tradeoffs=dominant_tradeoffs,
            limitations=tuple(item.message for item in self.limitations),
        )


def unavailable_decision_history_summary(reason: str) -> DecisionHistorySummary:
    return DecisionHistorySummary(
        schema_version=DECISION_HISTORY_SCHEMA_VERSION,
        available=False,
        selection_basis=None,
        candidate_count=None,
        winner_candidate_index=None,
        runner_up_candidate_index=None,
        score_margin_to_runner_up=None,
        dominant_tradeoffs=(),
        limitations=(reason,),
    )


def _candidate_lookup(trace: SearchTrace) -> dict[int, SearchCandidateEvaluation]:
    return {evaluation.index: evaluation for evaluation in trace.evaluations}


def _rank_lookup(trace: SearchTrace) -> dict[int, int]:
    return {index: rank for rank, index in enumerate(trace.ranked_candidate_indexes, start=1)}


def _candidate_summary(
    evaluation: SearchCandidateEvaluation,
    rank_lookup: dict[int, int],
) -> DecisionHistoryCandidateSummary:
    return DecisionHistoryCandidateSummary(
        index=evaluation.index,
        rank=rank_lookup[evaluation.index],
        params=evaluation.params,
        score=evaluation.score,
        objective_attribution=evaluation.attribution.to_dict(),
    )


def _term_lookup(evaluation: SearchCandidateEvaluation) -> dict[str, Any]:
    return {term.name: term for term in evaluation.attribution.terms}


def _separation_terms(
    winner: SearchCandidateEvaluation,
    alternative: SearchCandidateEvaluation,
    *,
    limit: int = 3,
) -> tuple[DecisionHistorySeparationTerm, ...]:
    winner_terms = _term_lookup(winner)
    alternative_terms = _term_lookup(alternative)
    terms: list[DecisionHistorySeparationTerm] = []
    for name, alt_term in alternative_terms.items():
        winner_term = winner_terms.get(name)
        if winner_term is None:
            continue
        penalty_delta = alt_term.penalty - winner_term.penalty
        if penalty_delta <= 1e-12:
            continue
        terms.append(
            DecisionHistorySeparationTerm(
                term=name,
                category=alt_term.category,
                winner_penalty=winner_term.penalty,
                alternative_penalty=alt_term.penalty,
                penalty_delta=penalty_delta,
                evidence_level="exact",
            )
        )
    terms.sort(key=lambda item: (-item.penalty_delta, item.term))
    return tuple(terms[:limit])


def _selection_tradeoffs(
    winner: SearchCandidateEvaluation,
    runner_up: SearchCandidateEvaluation | None,
) -> tuple[DecisionHistoryTradeoff, ...]:
    if runner_up is None:
        return ()
    separation_terms = _separation_terms(winner, runner_up)
    if separation_terms:
        top_names = ", ".join(item.term for item in separation_terms[:2])
        summary = (
            "Within the evaluated bounded grid, the winner beat the runner-up on total objective score, "
            f"with the largest separating penalties in {top_names}."
        )
    else:
        summary = (
            "Within the evaluated bounded grid, the winner beat the runner-up on total objective score, "
            "but no positive per-term penalty separation was recorded."
        )
    return (
        DecisionHistoryTradeoff(
            comparison_basis="winner_vs_runner_up",
            candidate_index=runner_up.index,
            score_delta_vs_winner=float(runner_up.score - winner.score),
            separation_terms=separation_terms,
            summary=summary,
            summary_level="inferred",
        ),
    )


def _reference_metric_tradeoffs(
    trace: SearchTrace,
    winner: SearchCandidateEvaluation,
    reference_available: bool,
) -> tuple[DecisionHistoryMetricTradeoff, ...]:
    if not reference_available:
        return ()
    winner_terms = _term_lookup(winner)
    tradeoffs: list[DecisionHistoryMetricTradeoff] = []
    for metric, term_name in REFERENCE_METRIC_TERM_NAMES.items():
        winner_term = winner_terms.get(term_name)
        if winner_term is None or winner_term.difference is None:
            tradeoffs.append(
                DecisionHistoryMetricTradeoff(
                    metric=metric,
                    status="unavailable",
                    evidence_level="unavailable",
                    winner_delta_to_reference=None,
                    alternative_candidate_index=None,
                    alternative_delta_to_reference=None,
                    alternative_score_delta_vs_winner=None,
                    blocking_terms=(),
                    summary="This run did not record enough evaluated evidence to compare this metric honestly.",
                    summary_level="unavailable",
                )
            )
            continue

        better_alternatives: list[tuple[float, float, SearchCandidateEvaluation]] = []
        for evaluation in trace.evaluations:
            if evaluation.index == winner.index:
                continue
            candidate_term = _term_lookup(evaluation).get(term_name)
            if candidate_term is None or candidate_term.difference is None:
                continue
            if abs(candidate_term.difference) + 1e-9 >= abs(winner_term.difference):
                continue
            better_alternatives.append((abs(candidate_term.difference), evaluation.score, evaluation))

        if not better_alternatives:
            tradeoffs.append(
                DecisionHistoryMetricTradeoff(
                    metric=metric,
                    status="no_closer_evaluated_candidate",
                    evidence_level="exact",
                    winner_delta_to_reference=float(winner_term.difference),
                    alternative_candidate_index=None,
                    alternative_delta_to_reference=None,
                    alternative_score_delta_vs_winner=None,
                    blocking_terms=(),
                    summary=(
                        "Within the evaluated bounded grid, no candidate landed closer to the reference on this metric "
                        "than the selected winner."
                    ),
                    summary_level="exact",
                )
            )
            continue

        _, _, alternative = min(better_alternatives, key=lambda item: (item[0], item[1], item[2].index))
        alternative_term = _term_lookup(alternative)[term_name]
        blocking_terms = _separation_terms(winner, alternative)
        tradeoffs.append(
            DecisionHistoryMetricTradeoff(
                metric=metric,
                status="closer_reference_match_rejected",
                evidence_level="exact",
                winner_delta_to_reference=float(winner_term.difference),
                alternative_candidate_index=alternative.index,
                alternative_delta_to_reference=float(alternative_term.difference),
                alternative_score_delta_vs_winner=float(alternative.score - winner.score),
                blocking_terms=blocking_terms,
                summary=(
                    "A closer reference match existed within the evaluated bounded grid, "
                    "but it lost on total objective score."
                ),
                summary_level="inferred",
            )
        )
    return tuple(tradeoffs)


def build_decision_history_artifact(
    trace: SearchTrace,
    targets: TargetValues,
) -> DecisionHistoryArtifact:
    lookup = _candidate_lookup(trace)
    rank_lookup = _rank_lookup(trace)
    winner = lookup[trace.best_index]
    runner_up = None if trace.runner_up_index is None else lookup[trace.runner_up_index]
    evaluated_candidates = tuple(
        _candidate_summary(evaluation, rank_lookup)
        for evaluation in sorted(trace.evaluations, key=lambda item: item.index)
    )
    competitive_alternatives = tuple(
        _candidate_summary(lookup[index], rank_lookup)
        for index in trace.ranked_candidate_indexes
        if index != trace.best_index
    )[:3]

    objective_schema_version = winner.attribution.schema_version
    limitations = (
        DecisionHistoryLimitation(
            code="BOUNDED_GRID_ONLY",
            message=(
                "This artifact covers all evaluated candidates in the bounded deterministic grid, "
                "not the full continuous parameter space."
            ),
            evidence_level="exact",
        ),
        DecisionHistoryLimitation(
            code="GLOBAL_OPTIMUM_UNAVAILABLE",
            message="The optimizer does not record evidence for a global optimum outside the evaluated bounded grid.",
            evidence_level="unavailable",
        ),
    )

    return DecisionHistoryArtifact(
        schema_version=DECISION_HISTORY_SCHEMA_VERSION,
        artifact_type="optimizer_decision_history",
        search=DecisionHistorySearch(
            search_trace_schema_version=trace.schema_version,
            objective_attribution_schema_version=objective_schema_version,
            selection_basis=trace.selection_basis,
            space_kind="bounded_grid",
            exhaustive_within_space=True,
            global_optimum_claim="unavailable",
            parameter_axes=trace.parameter_axes,
            candidate_count=trace.candidate_count,
            ranking_rule=trace.ranking_rule,
            best_tie_break_rule="first_lowest_score_by_candidate_index",
        ),
        context=DecisionHistoryContext(
            reference_available=targets.reference_targeting is not None,
            reference_targeting_schema_version=(
                None if targets.reference_targeting is None else targets.reference_targeting.schema_version
            ),
        ),
        selection=DecisionHistorySelection(
            winner_candidate_index=trace.best_index,
            runner_up_candidate_index=trace.runner_up_index,
            winner_score=trace.best_score,
            runner_up_score=trace.runner_up_score,
            score_margin_to_runner_up=trace.winner_score_margin_to_runner_up,
            tie_count_at_best_score=trace.tie_count_at_best_score,
        ),
        winner=_candidate_summary(winner, rank_lookup),
        runner_up=None if runner_up is None else _candidate_summary(runner_up, rank_lookup),
        competitive_alternatives=competitive_alternatives,
        selection_tradeoffs=_selection_tradeoffs(winner, runner_up),
        reference_metric_tradeoffs=_reference_metric_tradeoffs(
            trace,
            winner,
            targets.reference_targeting is not None,
        ),
        limitations=limitations,
        evaluated_candidates=evaluated_candidates,
    )
