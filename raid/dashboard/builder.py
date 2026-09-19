"""Visual builder model, and the compiler that turns it into DSL text.

Why this is a compiler and not a second parser
----------------------------------------------
The visual builder is strictly an *input layer*. It holds its own small form
model and emits DSL source text, which then goes through exactly the same
:func:`raid.dsl.parse` as anything typed by hand. Nothing downstream of
``raid/dsl`` knows the builder exists, and there is no second path into the
analysis - which is what keeps the DSL the single source of truth that CLAUDE.md
requires, and what makes "Generated DSL (this is what RAID actually parses)"
literally true rather than a claim.

That also means the builder can never express something the DSL cannot, and any
diagram it produces is reviewable as text before it is analysed.

Robustness
----------
:func:`compile_dsl` never emits text it knows will not parse. A half-filled
form, a blank method name, or an interaction left pointing at a service the
user has since deleted are all normal intermediate states in a live demo, so
each is skipped from the output and reported by :func:`validate` instead of
producing a syntax error the user cannot act on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

__all__ = [
    "LOGIN_DSL",
    "PARAMETER_TYPES",
    "VisualDiagram",
    "VisualFlow",
    "VisualInteraction",
    "VisualParameter",
    "compile_dsl",
    "login_example",
    "validate",
]

#: Types offered in the parameter dropdown. ``None`` is included because the
#: DSL permits an untyped parameter, and being able to build one is what lets
#: the sufficiency scorer's test-generation concern be demonstrated from the
#: visual builder rather than only from hand-written DSL.
PARAMETER_TYPES: tuple[str | None, ...] = ("int", "float", "string", "bool", None)

#: Words the grammar lexes as keywords; a service or flow named one of these
#: would not parse back.
_RESERVED = frozenset({"service", "flow", "alt", "else"})

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def _is_identifier(name: str) -> bool:
    """Whether a name can appear as a NAME token in the grammar."""
    return bool(name) and bool(_IDENTIFIER.match(name)) and name not in _RESERVED


@dataclass
class VisualParameter:
    """One parameter row in the form."""

    name: str = ""
    type: str | None = "string"


@dataclass
class VisualInteraction:
    """One call, plus the branch arm it belongs to.

    ``condition`` is ``None`` for a top-level call. When it is set, ``arm``
    decides whether this becomes the ``alt`` operand or the ``else`` operand -
    the two words the user is never asked to type.
    """

    source: str = ""
    target: str = ""
    method: str = ""
    parameters: list[VisualParameter] = field(default_factory=list)
    timeout_ms: int | None = None
    retry: int | None = None
    condition: str | None = None
    arm: str = "alt"


@dataclass
class VisualFlow:
    """A named flow and its ordered interactions."""

    name: str = ""
    interactions: list[VisualInteraction] = field(default_factory=list)


@dataclass
class VisualDiagram:
    """The whole form state: declared services plus flows."""

    services: list[str] = field(default_factory=list)
    flows: list[VisualFlow] = field(default_factory=list)

    def flow(self, name: str) -> VisualFlow | None:
        for flow in self.flows:
            if flow.name == name:
                return flow
        return None


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------


def _emit_parameters(parameters: list[VisualParameter]) -> str:
    rendered = []
    for parameter in parameters:
        if not _is_identifier(parameter.name):
            continue
        rendered.append(
            f"{parameter.name}:{parameter.type}" if parameter.type else parameter.name
        )
    return ", ".join(rendered)


def _emit_annotations(interaction: VisualInteraction) -> str:
    parts = []
    if interaction.timeout_ms is not None:
        # Always emitted in milliseconds: the form collects ms, and `2s` and
        # `2000ms` are the same value to the simulation's duration parser.
        parts.append(f"timeout={interaction.timeout_ms:g}ms")
    if interaction.retry is not None:
        parts.append(f"retry={interaction.retry:g}")
    return f" [{', '.join(parts)}]" if parts else ""


def _emit_interaction(interaction: VisualInteraction) -> str:
    return (
        f"{interaction.source} -> {interaction.target} : "
        f"{interaction.method}({_emit_parameters(interaction.parameters)})"
        f"{_emit_annotations(interaction)}"
    )


def _is_emittable(interaction: VisualInteraction) -> bool:
    """Whether this interaction can be written as parseable DSL."""
    return all(
        _is_identifier(value)
        for value in (interaction.source, interaction.target, interaction.method)
    )


def _group(interactions: list[VisualInteraction]):
    """Collapse consecutive interactions sharing a branch arm into one block.

    Grouping is positional: a run of interactions carrying the same condition
    and arm becomes one ``alt``/``else`` operand. That is what lets the user
    build a branch by tagging individual rows, without ever being shown the
    block structure they imply.
    """
    groups: list[tuple[str | None, str, list[VisualInteraction]]] = []
    for interaction in interactions:
        # A condition that is not a usable name would emit `alt bad name:`,
        # which does not parse. Drop the branch and keep the call at the top
        # level rather than losing the interaction the user actually built.
        condition = interaction.condition or None
        if condition is not None and not _is_identifier(condition):
            condition = None
        arm = interaction.arm if condition else "alt"
        if groups and groups[-1][0] == condition and groups[-1][1] == arm:
            groups[-1][2].append(interaction)
        else:
            groups.append((condition, arm, [interaction]))
    return groups


def _emit_flow_body(interactions: list[VisualInteraction]) -> list[str]:
    lines: list[str] = []
    emittable = [i for i in interactions if _is_emittable(i)]

    previous_was_alt = False
    for condition, arm, members in _group(emittable):
        if condition is None:
            lines.extend(f"  {_emit_interaction(i)}" for i in members)
            previous_was_alt = False
            continue

        # An `else` only exists as the second operand of an `alt`. If the user
        # tagged a row as the alternative arm with no main arm before it, emit
        # it as its own `alt` so the text still parses; validate() explains.
        keyword = "else" if arm == "else" and previous_was_alt else "alt"
        lines.append(f"  {keyword} {condition}:")
        lines.extend(f"    {_emit_interaction(i)}" for i in members)
        previous_was_alt = keyword == "alt"

    return lines


def compile_dsl(diagram: VisualDiagram) -> str:
    """Render the form state as DSL source text.

    Anything that would not parse - a blank name, a reserved word, an
    interaction still missing its method - is omitted rather than written out
    broken. The result is always valid DSL, possibly describing less than the
    user has typed so far; :func:`validate` reports what was left out.

    Args:
        diagram: The current builder state.

    Returns:
        DSL text, newline terminated. An empty model compiles to ``""``.
    """
    blocks: list[str] = []

    services = [s for s in diagram.services if _is_identifier(s)]
    if services:
        blocks.append("\n".join(f"service {name}" for name in services))

    for flow in diagram.flows:
        if not _is_identifier(flow.name):
            continue
        body = _emit_flow_body(flow.interactions)
        if not body:
            continue
        blocks.append("\n".join([f"flow {flow.name}:", *body]))

    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate(diagram: VisualDiagram) -> list[str]:
    """Return human-readable warnings about the current form state.

    These are advisory, never fatal: the builder is expected to spend most of
    its life in a half-finished state, and a live demo should be told what is
    missing rather than shown an error.
    """
    warnings: list[str] = []
    declared = [s for s in diagram.services if _is_identifier(s)]

    for name in diagram.services:
        if not _is_identifier(name):
            warnings.append(
                f"Service name {name!r} is not usable: use letters, digits and "
                f"underscores only, starting with a letter, and avoid the words "
                f"service/flow/alt/else. It has been left out of the DSL."
            )

    if len(set(declared)) != len(declared):
        duplicates = sorted({s for s in declared if declared.count(s) > 1})
        warnings.append(f"Declared more than once: {', '.join(duplicates)}.")

    if not declared:
        warnings.append("No services yet — add one to get started.")

    if not diagram.flows:
        warnings.append("No flows yet — add a flow, then add interactions to it.")

    known = set(declared)
    for flow in diagram.flows:
        if not _is_identifier(flow.name):
            warnings.append(
                f"Flow name {flow.name!r} is not usable and has been left out of the DSL."
            )
            continue

        if not flow.interactions:
            warnings.append(f"Flow {flow.name!r} has no interactions yet.")
            continue

        previous_was_alt = False
        for position, interaction in enumerate(flow.interactions, start=1):
            # Built pre-capitalised: str.capitalize() would lowercase the rest
            # of the string and corrupt the case-sensitive names inside it.
            label = f"interaction {position} in {flow.name!r}"
            sentence_label = "I" + label[1:]

            if not _is_emittable(interaction):
                warnings.append(
                    f"{sentence_label} is incomplete (source, target and method "
                    f"must all be filled in) and has been left out of the DSL."
                )
                continue

            for role, name in (("source", interaction.source), ("target", interaction.target)):
                if name not in known:
                    warnings.append(
                        f"{sentence_label} has {role} {name!r}, which is not a "
                        f"declared service — it was probably deleted. RAID will still "
                        f"analyse it, and the sufficiency scorer will flag it."
                    )

            for parameter in interaction.parameters:
                if parameter.name and not _is_identifier(parameter.name):
                    warnings.append(
                        f"Parameter {parameter.name!r} in {label} is not a usable name "
                        f"and has been left out."
                    )

            condition = interaction.condition or None
            if condition and not _is_identifier(condition):
                warnings.append(
                    f"Condition {condition!r} on {label} is not a usable name; "
                    f"the branch has been left out of the DSL."
                )
            if condition and interaction.arm == "else" and not previous_was_alt:
                warnings.append(
                    f"{sentence_label} is marked as the alternative arm, but no "
                    f"main branch comes before it. It has been written as its own "
                    f"branch instead."
                )
            previous_was_alt = bool(condition) and interaction.arm == "alt"

    return warnings


# --------------------------------------------------------------------------
# The worked example the builder opens with
# --------------------------------------------------------------------------

#: The canonical Login diagram, shared by both input modes so the visual
#: builder and the code editor open showing exactly the same thing.
#: ``compile_dsl(login_example())`` reproduces this byte for byte, which is
#: asserted in the tests.
LOGIN_DSL = """service Client
service AuthService
service UserStore

flow Login:
  Client -> AuthService : authenticate(email:string, password:string) [timeout=1000ms, retry=2]
  AuthService -> UserStore : loadProfile(userId:int) [timeout=500ms, retry=3]
  AuthService -> Client : issueToken(sessionId:string) [timeout=200ms, retry=1]
"""


def login_example() -> VisualDiagram:
    """Build the Login diagram as builder state, so the form opens populated."""
    return VisualDiagram(
        services=["Client", "AuthService", "UserStore"],
        flows=[
            VisualFlow(
                name="Login",
                interactions=[
                    VisualInteraction(
                        source="Client",
                        target="AuthService",
                        method="authenticate",
                        parameters=[
                            VisualParameter("email", "string"),
                            VisualParameter("password", "string"),
                        ],
                        timeout_ms=1000,
                        retry=2,
                    ),
                    VisualInteraction(
                        source="AuthService",
                        target="UserStore",
                        method="loadProfile",
                        parameters=[VisualParameter("userId", "int")],
                        timeout_ms=500,
                        retry=3,
                    ),
                    VisualInteraction(
                        source="AuthService",
                        target="Client",
                        method="issueToken",
                        parameters=[VisualParameter("sessionId", "string")],
                        timeout_ms=200,
                        retry=1,
                    ),
                ],
            )
        ],
    )
