"""Exception types raised by the RAID DSL layer.

Every failure the DSL layer surfaces is one of these, so callers (the CLI, the
Streamlit dashboard, other RAID modules) can catch a single well-known type and
show the message verbatim. Lark's own exceptions are never allowed to escape:
they carry parser-internal detail (rule names, LALR states) that is meaningless
to someone writing a diagram.
"""


class DslError(Exception):
    """Base class for all errors raised by the RAID DSL layer."""


class DslSyntaxError(DslError):
    """Raised when DSL source text cannot be parsed.

    The message is intended to be shown directly to the user: it names the
    location, what was found there, and what the parser expected instead.
    """
