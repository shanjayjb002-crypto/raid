"""Parser for the RAID architecture DSL.

Pipeline
--------
``source text -> lark tokens -> _DslIndenter -> LALR parse tree -> _AstBuilder
-> Diagram``

The parse is a two-stage design. Lark builds a generic tree, then
:class:`_AstBuilder` lowers that tree into the dataclasses in
:mod:`raid.dsl.model`. Keeping the lowering separate (rather than passing the
transformer straight to ``Lark(transformer=...)``) costs one extra tree walk but
means a malformed document fails during parsing, where full position
information is still available for error messages.

LALR(1) is used rather than Earley because the grammar is unambiguous and LALR
gives both faster parsing and - more importantly here - precise, single-token
error reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lark import Lark, Token, Transformer
from lark.exceptions import LarkError, UnexpectedCharacters, UnexpectedInput, UnexpectedToken
from lark.indenter import Indenter

from .errors import DslSyntaxError
from .model import (
    AnnotationValue,
    Branch,
    Diagram,
    Flow,
    Interaction,
    Parameter,
    Service,
    Step,
)

__all__ = ["parse"]

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


class _DslIndenter(Indenter):
    """Converts leading whitespace into ``_INDENT``/``_DEDENT`` tokens.

    Lark's lexer has no notion of block structure, so this post-lexer supplies
    it: it tracks a stack of indentation columns and emits a token whenever the
    level changes. Indentation inside parentheses is ignored, so a long
    parameter list could be wrapped across lines without being mistaken for a
    new block.

    Any dedent that does not return to a column already on the stack raises
    lark's ``DedentError``, which :func:`parse` reports as a normal syntax
    error - that is what catches misaligned ``alt``/``else`` bodies.
    """

    NL_type = "_NL"
    OPEN_PAREN_types = ["LPAR"]
    CLOSE_PAREN_types = ["RPAR"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4


@dataclass
class _ElseClause:
    """Internal carrier for a parsed ``else`` arm.

    Only exists so ``branch`` can tell its own steps apart from its else arm's
    steps while walking children; it never escapes this module.
    """

    condition: str
    steps: list[Step]


def _coerce_annotation_value(token: Token) -> AnnotationValue:
    """Convert an annotation's value token into a plain Python value.

    Coercion is driven by the terminal that matched, not by re-inspecting the
    text, so the grammar stays the single source of truth about what each
    literal form means.
    """
    text = str(token)
    if token.type == "SIGNED_NUMBER":
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    if token.type == "ESCAPED_STRING":
        return text[1:-1]
    # DURATION and bare NAME values are kept verbatim; see model.AnnotationValue.
    return text


class _AstBuilder(Transformer):
    """Lowers a lark parse tree into the :mod:`raid.dsl.model` dataclasses.

    Each method is named after a grammar rule and receives that rule's already
    transformed children, so the tree is rebuilt bottom-up.
    """

    def start(self, children: list) -> Diagram:
        return Diagram(
            services=[c for c in children if isinstance(c, Service)],
            flows=[c for c in children if isinstance(c, Flow)],
        )

    def service_decl(self, children: list) -> Service:
        (name,) = children
        return Service(name=str(name), line=name.line)

    def flow_decl(self, children: list) -> Flow:
        name, *steps = children
        return Flow(name=str(name), steps=steps, line=name.line)

    def interaction(self, children: list) -> Interaction:
        source, target, method, *optional = children

        # `param_list?` and `annotation_block?` are both optional, so the tail
        # is matched on the type each rule produces rather than on position.
        parameters: list[Parameter] = []
        annotations: dict[str, AnnotationValue] = {}
        for extra in optional:
            if isinstance(extra, list):
                parameters = extra
            elif isinstance(extra, dict):
                annotations = extra

        return Interaction(
            source=str(source),
            target=str(target),
            method=str(method),
            parameters=parameters,
            annotations=annotations,
            line=source.line,
        )

    def param_list(self, children: list) -> list[Parameter]:
        return list(children)

    def param(self, children: list) -> Parameter:
        name, type_name = children
        return Parameter(name=str(name), type=str(type_name))

    def branch(self, children: list) -> Branch:
        condition, *rest = children

        steps: list[Step] = []
        else_condition: str | None = None
        else_steps: list[Step] = []
        for child in rest:
            if isinstance(child, _ElseClause):
                else_condition = child.condition
                else_steps = child.steps
            else:
                steps.append(child)

        return Branch(
            condition=str(condition),
            steps=steps,
            else_condition=else_condition,
            else_steps=else_steps,
            line=condition.line,
        )

    def else_clause(self, children: list) -> _ElseClause:
        condition, *steps = children
        return _ElseClause(condition=str(condition), steps=steps)

    def annotation_block(self, children: list) -> dict[str, AnnotationValue]:
        return dict(children)

    def annotation(self, children: list) -> tuple[str, AnnotationValue]:
        name, value = children
        return str(name), _coerce_annotation_value(value)


_parser = Lark(
    _GRAMMAR_PATH.read_text(encoding="utf-8"),
    parser="lalr",
    postlex=_DslIndenter(),
    start="start",
)


# Readable stand-ins for terminals whose names would mean nothing to a diagram
# author. String terminals (keywords, punctuation) are described automatically
# from their own pattern, so only the regex ones need naming here.
_TERMINAL_LABELS: dict[str, str] = {
    "NAME": "a name",
    "SIGNED_NUMBER": "a number",
    "DURATION": "a duration such as 2s",
    "ESCAPED_STRING": "a quoted string",
    "_NL": "a line break",
    "_INDENT": "an indented block",
    "_DEDENT": "the end of an indented block",
    "$END": "end of input",
}


def _describe_terminal(name: str) -> str:
    """Render one expected-terminal name in language a diagram author can act on."""
    if name in _TERMINAL_LABELS:
        return _TERMINAL_LABELS[name]
    for terminal in _parser.terminals:
        if terminal.name == name and terminal.pattern.type == "str":
            return f"'{terminal.pattern.value}'"
    return name


def _describe_expected(names: object, limit: int = 6) -> str:
    """Summarise the set of terminals the parser would have accepted."""
    if not names:
        return ""
    described = sorted({_describe_terminal(str(n)) for n in names})  # type: ignore[union-attr]
    if len(described) > limit:
        described = described[:limit] + ["..."]
    return ", ".join(described)


def _describe_found(token: Token) -> str:
    """Describe the token that actually turned up at the error position."""
    text = str(token)
    if token.type == "$END":
        return "end of input"
    if not text:
        # _INDENT/_DEDENT are synthesised by the post-lexer and carry no source
        # text, so quoting their (empty) value would only add noise.
        return _describe_terminal(token.type)
    if token.type in _TERMINAL_LABELS:
        return f"{_describe_terminal(token.type)} ({text!r})"
    return repr(text)


def _format_syntax_error(exc: UnexpectedInput, text: str) -> str:
    """Build the user-facing message for a lark parse failure.

    Deliberately reconstructs the message instead of reusing ``str(exc)``: lark
    reports LALR state numbers and internal rule names, which are noise to
    someone who just mistyped an arrow.
    """
    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)
    # Tokens the post-lexer synthesises have no position, which happens when a
    # construct is left unterminated at the end of a block.
    location = f"line {line}, column {column}" if line else "the end of the document"

    if isinstance(exc, UnexpectedToken):
        found = _describe_found(exc.token)
        # `accepts` is the set the LALR table would actually shift next;
        # `expected` is wider and can name a terminal that would not in fact
        # help (it suggests a line break where an indented block is required).
        expected = _describe_expected(exc.accepts or exc.expected)
    elif isinstance(exc, UnexpectedCharacters):
        position = getattr(exc, "pos_in_stream", None)
        character = text[position] if position is not None and position < len(text) else "?"
        found = f"unrecognised character {character!r}"
        expected = _describe_expected(getattr(exc, "allowed", None))
    else:
        found = "unexpected input"
        expected = _describe_expected(getattr(exc, "expected", None))

    message = f"RAID DSL syntax error at {location}: found {found}."
    if expected:
        message += f" Expected one of: {expected}."

    try:
        context = exc.get_context(text).rstrip()
    except Exception:  # pragma: no cover - context is a convenience, never critical
        context = ""
    if context:
        message += "\n\n" + context

    return message


def parse(source: str) -> Diagram:
    """Parse RAID DSL source text into a :class:`~raid.dsl.model.Diagram`.

    Args:
        source: The full text of a DSL document.

    Returns:
        The parsed diagram. An empty document yields an empty diagram rather
        than an error, so the dashboard can render a blank editor without
        special-casing it.

    Raises:
        DslSyntaxError: If the text is not valid DSL. The message names the
            line, column, what was found and what was expected, and is safe to
            show directly to the user.
    """
    # The grammar terminates every statement with a newline, so a document
    # whose last line lacks one is completed here rather than being reported
    # as a confusing "unexpected end of input".
    text = source if source.endswith("\n") else source + "\n"

    try:
        tree = _parser.parse(text)
    except UnexpectedInput as exc:
        raise DslSyntaxError(_format_syntax_error(exc, text)) from None
    except LarkError as exc:
        # Chiefly DedentError from the indenter: the text lexes but its block
        # structure does not line up.
        raise DslSyntaxError(f"RAID DSL syntax error: {exc}") from None

    return _AstBuilder().transform(tree)
