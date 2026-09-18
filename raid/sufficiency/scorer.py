"""Scores how analysable a diagram is, before any generation is attempted.

Why score before generating
---------------------------
Every later phase degrades quietly on an under-specified diagram rather than
failing: an untyped parameter still gets boundary cases (inferred from its name
alone, with visibly lower confidence), an unannotated call still simulates
(against a default timeout the author never chose), an uncalled service still
appears in the graph (but no mutation of it can bite). None of that is wrong,
but all of it is *guessing*, and the guesses are invisible in the output.

This module makes them visible up front, so an analyst knows which results rest
on their diagram and which rest on RAID's defaults.

How each concern is judged
--------------------------
Each concern is scored in the unit it actually consumes, which keeps the
fraction meaningful rather than arbitrary:

* **testgen** counts *parameters*, because a parameter is what a boundary case
  is generated from. An untyped one is still usable but unreliable.
* **resilience** counts *annotation slots* - a timeout and a retry per
  interaction - because those are what the simulation reads. Scoring on
  timeouts alone would call a half-annotated diagram fully ready.
* **fuzzing** counts *services*, because a service is the unit a mutation
  targets, and one that never receives a call cannot be killed to any effect.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from raid.dsl.model import Branch, Interaction, Step
from raid.graph import RaidGraph
from raid.trace import interaction_id, service_id

from .model import CONCERNS, SCORES, ConcernScore, SufficiencyReport, Suggestion

__all__ = ["score_sufficiency"]

#: A concern is "high" once this share of its units are ready, "medium" at the
#: lower bound, "low" below it. The thresholds are a judgement about when an
#: analyst can trust a result, so they live here rather than being buried.
_HIGH_THRESHOLD = 0.9
_MEDIUM_THRESHOLD = 0.5

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _walk(steps: Sequence[Step]) -> Iterable[Interaction]:
    """Yield every interaction in a step tree, both arms of every branch."""
    for step in steps:
        if isinstance(step, Interaction):
            yield step
        elif isinstance(step, Branch):
            yield from _walk(step.steps)
            yield from _walk(step.else_steps)


def _grade(ready: int, total: int) -> tuple[str, float]:
    """Turn a ready/total count into a score band.

    A concern with nothing to judge scores ``low``: an empty diagram is not
    vacuously analysable, it is simply not analysable.
    """
    if total == 0:
        return "low", 0.0
    ratio = ready / total
    if ratio >= _HIGH_THRESHOLD:
        return "high", ratio
    if ratio >= _MEDIUM_THRESHOLD:
        return "medium", ratio
    return "low", ratio


def _signature(interaction: Interaction) -> str:
    return f"{interaction.source} -> {interaction.target} : {interaction.method}"


def _score_testgen(
    interactions: list[tuple[str, Interaction]],
) -> tuple[ConcernScore, list[Suggestion]]:
    """Flag parameters with no declared type."""
    suggestions: list[Suggestion] = []
    typed = 0
    total = 0

    for flow_name, interaction in interactions:
        for parameter in interaction.parameters:
            total += 1
            if parameter.type is not None:
                typed += 1
                continue
            suggestions.append(
                Suggestion(
                    concern="testgen",
                    severity="high",
                    element=_signature(interaction),
                    element_id=interaction_id(flow_name, interaction.line),
                    message=(
                        f"add a type to {parameter.name!r} in "
                        f"{interaction.method}() to enable boundary test generation "
                        f"(currently inferred from the name alone)"
                    ),
                )
            )

    score, ratio = _grade(typed, total)
    return (
        ConcernScore(
            concern="testgen",
            score=score,
            ratio=ratio,
            ready=typed,
            total=total,
            unit="parameters",
            rationale=f"{typed} of {total} parameters carry a declared type",
        ),
        suggestions,
    )


def _score_resilience(
    interactions: list[tuple[str, Interaction]],
) -> tuple[ConcernScore, list[Suggestion]]:
    """Flag interactions missing the annotations the simulation reads."""
    suggestions: list[Suggestion] = []
    annotated = 0
    total = 0

    for flow_name, interaction in interactions:
        total += 2  # one slot for timeout, one for retry
        element = _signature(interaction)
        node = interaction_id(flow_name, interaction.line)

        if "timeout" in interaction.annotations:
            annotated += 1
        else:
            suggestions.append(
                Suggestion(
                    concern="resilience",
                    severity="high",
                    element=element,
                    element_id=node,
                    message=(
                        f"add [timeout=...] to {interaction.method}() in flow "
                        f"{flow_name} to enable resilience simulation "
                        f"(currently falls back to the default budget)"
                    ),
                )
            )

        if "retry" in interaction.annotations:
            annotated += 1
        else:
            suggestions.append(
                # Lower severity than a missing timeout: without a timeout the
                # simulation has to invent a budget, whereas without a retry it
                # merely models one attempt.
                Suggestion(
                    concern="resilience",
                    severity="low",
                    element=element,
                    element_id=node,
                    message=(
                        f"add [retry=...] to {interaction.method}() in flow "
                        f"{flow_name} to describe its recovery behaviour"
                    ),
                )
            )

    score, ratio = _grade(annotated, total)
    return (
        ConcernScore(
            concern="resilience",
            score=score,
            ratio=ratio,
            ready=annotated,
            total=total,
            unit="annotation slots",
            rationale=f"{annotated} of {total} timeout/retry annotations present",
        ),
        suggestions,
    )


def _score_fuzzing(graph: RaidGraph) -> tuple[ConcernScore, list[Suggestion]]:
    """Flag services that no mutation could meaningfully target."""
    suggestions: list[Suggestion] = []
    mutable = 0
    services = sorted(graph.graph.nodes)

    for service in services:
        node = service_id(service)
        incoming = graph.graph.in_degree(service)
        outgoing = graph.graph.out_degree(service)
        declared = graph.graph.nodes[service].get("declared", True)

        if not declared:
            suggestions.append(
                Suggestion(
                    concern="fuzzing",
                    severity="high",
                    element=service,
                    element_id=node,
                    message=(
                        f"{service!r} is called but never declared; add "
                        f"'service {service}' so it can be modelled and mutated"
                    ),
                )
            )

        if incoming == 0 and outgoing == 0:
            suggestions.append(
                Suggestion(
                    concern="fuzzing",
                    severity="high",
                    element=service,
                    element_id=node,
                    message=(
                        f"{service!r} has no incoming or outgoing calls; "
                        f"nothing to mutate - either wire it into a flow or "
                        f"remove the declaration"
                    ),
                )
            )
        elif incoming == 0:
            # Established in the simulation phase: failing a service that only
            # originates calls changes nothing, because no call ever waits on it.
            suggestions.append(
                Suggestion(
                    concern="fuzzing",
                    severity="medium",
                    element=service,
                    element_id=node,
                    message=(
                        f"{service!r} never receives a call, so killing it would "
                        f"change nothing; add an interaction targeting it to make "
                        f"it mutable"
                    ),
                )
            )
        elif declared:
            mutable += 1

    score, ratio = _grade(mutable, len(services))
    return (
        ConcernScore(
            concern="fuzzing",
            score=score,
            ratio=ratio,
            ready=mutable,
            total=len(services),
            unit="services",
            rationale=f"{mutable} of {len(services)} services can be meaningfully mutated",
        ),
        suggestions,
    )


def score_sufficiency(graph: RaidGraph) -> SufficiencyReport:
    """Score how analysable a diagram is, per concern, before generating anything.

    Args:
        graph: The graph model of a parsed diagram.

    Returns:
        A :class:`SufficiencyReport` with one score per concern and a list of
        specific changes, most severe first. A well-specified diagram yields no
        suggestions at all.
    """
    interactions = [
        (flow_name, interaction)
        for flow_name, flow in graph.flows.items()
        for interaction in _walk(flow.steps)
    ]

    scores = []
    suggestions: list[Suggestion] = []
    for concern_score, concern_suggestions in (
        _score_testgen(interactions),
        _score_resilience(interactions),
        _score_fuzzing(graph),
    ):
        scores.append(concern_score)
        suggestions.extend(concern_suggestions)

    suggestions.sort(
        key=lambda s: (_SEVERITY_ORDER[s.severity], CONCERNS.index(s.concern), s.element_id)
    )
    return SufficiencyReport(concerns=tuple(scores), suggestions=tuple(suggestions))
