"""Boundary-value inference for DSL parameters.

What this does
--------------
Given a parameter as written in a diagram (``amount:float``, ``cardId:string``)
it infers a *boundary class* and proposes concrete edge-case values for it.

Design notes (for the viva)
---------------------------
**Why a classifier and not a lookup table.** A hard-coded name->class map only
works for names it has seen. A TF-IDF model over *sub-word tokens* generalises:
having learned that the token ``id`` indicates an identifier and ``total``
indicates a quantity, it classifies ``customerId`` and ``basketTotal`` correctly
without either appearing in training. That generalisation is the whole point,
so the tokeniser matters more than the model does.

**Why the declared type is a feature, not an afterthought.** A boundary class
describes a *value space*, not a semantic role. ``itemId:int`` and
``cardId:string`` are both identifiers by name, but their boundaries have
nothing in common: the first has zero/negative/max, the second has
empty/malformed. Feeding the declared type in as its own token lets one model
make that distinction, and gives a useful fallback for free - a name the model
has never seen still carries its ``type=int`` token, so it lands on ``numeric``
rather than on an arbitrary class.

**Why logistic regression.** The feature space is small, sparse and nearly
linearly separable, so a linear model is sufficient; more importantly it is
*inspectable*. ``explain()`` reads the learned coefficients back out, so any
classification can be justified by naming the tokens that drove it - which a
tree ensemble or a neural model would not make nearly as easy to defend.

**No external services.** Training data is hand-built and lives in this file;
the model trains locally in milliseconds on import of the first use. Nothing
leaves the machine.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from raid.dsl.model import Parameter

__all__ = [
    "BOUNDARY_CASES",
    "TRAINING_EXAMPLES",
    "BoundaryCandidate",
    "BoundaryProposal",
    "ParameterClassifier",
    "get_classifier",
]


# --------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------
#
# Hand-built, deliberately small, and deliberately varied in surface form
# (camelCase, snake_case, abbreviations) so the tokeniser has to earn its keep.
# Each entry is (parameter name, declared type, boundary class).
#
# The type column is what separates `itemId:int` (numeric) from
# `cardId:string` (identifier) - see the module docstring.

TRAINING_EXAMPLES: list[tuple[str, str, str]] = [
    # --- numeric: things measured or counted ------------------------------
    ("amount", "float", "numeric"),
    ("qty", "int", "numeric"),
    ("quantity", "int", "numeric"),
    ("count", "int", "numeric"),
    ("total", "float", "numeric"),
    ("price", "float", "numeric"),
    ("unitPrice", "float", "numeric"),
    ("balance", "float", "numeric"),
    ("subtotal", "float", "numeric"),
    ("discount", "float", "numeric"),
    ("tax", "float", "numeric"),
    ("fee", "float", "numeric"),
    ("rate", "float", "numeric"),
    ("percentage", "float", "numeric"),
    ("weight", "float", "numeric"),
    ("height", "float", "numeric"),
    ("width", "float", "numeric"),
    ("age", "int", "numeric"),
    ("size", "int", "numeric"),
    ("limit", "int", "numeric"),
    ("offset", "int", "numeric"),
    ("page", "int", "numeric"),
    ("score", "float", "numeric"),
    ("retries", "int", "numeric"),
    ("attempts", "int", "numeric"),
    ("stock_level", "int", "numeric"),
    ("threshold", "float", "numeric"),
    ("capacity", "int", "numeric"),
    ("duration", "int", "numeric"),
    ("timeoutSeconds", "int", "numeric"),
    # Integer-typed identifiers belong here: their value space is the integers.
    ("itemId", "int", "numeric"),
    ("orderId", "int", "numeric"),
    ("userId", "int", "numeric"),
    ("customerId", "int", "numeric"),
    ("productId", "int", "numeric"),
    ("accountId", "int", "numeric"),
    ("id", "int", "numeric"),
    # --- identifier: opaque string keys -----------------------------------
    ("cardId", "string", "identifier"),
    ("orderId", "string", "identifier"),
    ("userId", "string", "identifier"),
    ("customerId", "string", "identifier"),
    ("itemId", "string", "identifier"),
    ("sessionId", "string", "identifier"),
    ("transactionId", "string", "identifier"),
    ("correlationId", "string", "identifier"),
    ("uuid", "string", "identifier"),
    ("guid", "string", "identifier"),
    ("sku", "string", "identifier"),
    ("reference", "string", "identifier"),
    ("referenceCode", "string", "identifier"),
    ("trackingNumber", "string", "identifier"),
    ("accountNumber", "string", "identifier"),
    ("invoiceNumber", "string", "identifier"),
    ("postcode", "string", "identifier"),
    ("countryCode", "string", "identifier"),
    ("currencyCode", "string", "identifier"),
    ("token", "string", "identifier"),
    ("apiKey", "string", "identifier"),
    ("slug", "string", "identifier"),
    # --- email ------------------------------------------------------------
    ("email", "string", "email"),
    ("emailAddress", "string", "email"),
    ("userEmail", "string", "email"),
    ("customerEmail", "string", "email"),
    ("contactEmail", "string", "email"),
    ("recipientEmail", "string", "email"),
    ("senderEmail", "string", "email"),
    ("billingEmail", "string", "email"),
    ("email_address", "string", "email"),
    ("notificationEmail", "string", "email"),
    # --- text: free-form human-written strings ----------------------------
    ("name", "string", "text"),
    ("firstName", "string", "text"),
    ("lastName", "string", "text"),
    ("fullName", "string", "text"),
    ("description", "string", "text"),
    ("title", "string", "text"),
    ("comment", "string", "text"),
    ("note", "string", "text"),
    ("notes", "string", "text"),
    ("message", "string", "text"),
    ("address", "string", "text"),
    ("street", "string", "text"),
    ("city", "string", "text"),
    ("label", "string", "text"),
    ("summary", "string", "text"),
    ("reason", "string", "text"),
    ("remark", "string", "text"),
    ("body", "string", "text"),
    ("content", "string", "text"),
    ("displayName", "string", "text"),
    ("nickname", "string", "text"),
    ("company", "string", "text"),
    # --- timestamp --------------------------------------------------------
    ("createdAt", "string", "timestamp"),
    ("updatedAt", "string", "timestamp"),
    ("deletedAt", "string", "timestamp"),
    ("timestamp", "string", "timestamp"),
    ("date", "string", "timestamp"),
    ("orderDate", "string", "timestamp"),
    ("expiryDate", "string", "timestamp"),
    ("dueDate", "string", "timestamp"),
    ("birthDate", "string", "timestamp"),
    ("scheduledTime", "string", "timestamp"),
    ("startTime", "string", "timestamp"),
    ("endTime", "string", "timestamp"),
    # --- boolean ----------------------------------------------------------
    ("isActive", "bool", "boolean"),
    ("isValid", "bool", "boolean"),
    ("enabled", "bool", "boolean"),
    ("confirmed", "bool", "boolean"),
    ("hasDiscount", "bool", "boolean"),
    ("isDeleted", "bool", "boolean"),
    ("flag", "bool", "boolean"),
    ("force", "bool", "boolean"),
    ("dryRun", "bool", "boolean"),
    ("verified", "bool", "boolean"),
]


#: The boundary cases proposed for each class, in the order they are generated.
#: Keeping this separate from the model means the *what to test* decision stays
#: reviewable by a human, while only the *which class* decision is learned.
BOUNDARY_CASES: dict[str, tuple[str, ...]] = {
    "numeric": ("zero", "negative", "max_value"),
    "identifier": ("empty", "malformed", "unknown"),
    "email": ("empty", "malformed", "very_long"),
    "text": ("empty", "whitespace_only", "very_long"),
    "timestamp": ("epoch", "far_past", "far_future"),
    "boolean": ("true", "false"),
}

_INT_TYPES = {"int", "integer", "long", "short", "number"}
_FLOAT_TYPES = {"float", "double", "decimal", "real"}

_MAX_INT = 2**31 - 1


@dataclass(frozen=True)
class BoundaryCandidate:
    """One proposed edge case: a named boundary and a concrete value for it."""

    case: str
    value: object


@dataclass(frozen=True)
class BoundaryProposal:
    """The classifier's verdict on one parameter, plus the cases it implies."""

    parameter: Parameter
    inferred_class: str
    confidence: float
    candidates: tuple[BoundaryCandidate, ...]


_SPLIT_PATTERN = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])")


def tokenize(name: str) -> list[str]:
    """Split a parameter name into lowercase sub-word tokens.

    ``cardId`` -> ``['card', 'id']``, ``stock_level`` -> ``['stock', 'level']``,
    ``timeoutSeconds`` -> ``['timeout', 'seconds']``.

    This is what lets the model generalise. Without it, TF-IDF would treat
    ``cardId`` and ``orderId`` as two unrelated whole-word features and the
    model could only ever recognise names it had already seen.
    """
    return [token.lower() for token in _SPLIT_PATTERN.split(name) if token]


def _document(name: str, declared_type: str | None) -> str:
    """Render a parameter as the text document the vectoriser consumes.

    The declared type is emitted as its own ``type=<t>`` token so it cannot be
    confused with a name token that happens to share its spelling (a parameter
    genuinely called ``string`` would otherwise be indistinguishable from the
    type ``string``).

    An untyped parameter emits ``type=unknown``, a token absent from the
    training vocabulary and therefore ignored by the vectoriser. The inference
    then rests on the name alone, which still works but with visibly lower
    confidence - exactly the signal the sufficiency scorer flags.
    """
    marker = (declared_type or "unknown").lower()
    return " ".join([*tokenize(name), f"type={marker}"])


def _numeric_value(case: str, declared_type: str | None) -> object:
    """Render a numeric boundary in the parameter's own declared type.

    An untyped parameter falls back to integer boundaries, the safer default:
    an int value is accepted anywhere a float is, but not the reverse.
    """
    is_float = (declared_type or "").lower() in _FLOAT_TYPES
    if case == "zero":
        return 0.0 if is_float else 0
    if case == "negative":
        return -1.0 if is_float else -1
    return sys.float_info.max if is_float else _MAX_INT


_STATIC_VALUES: dict[tuple[str, str], object] = {
    ("identifier", "empty"): "",
    ("identifier", "malformed"): "!!! not a valid id !!!",
    ("identifier", "unknown"): "00000000-0000-0000-0000-000000000000",
    ("email", "empty"): "",
    ("email", "malformed"): "not-an-email",
    ("email", "very_long"): "a" * 320 + "@example.com",
    ("text", "empty"): "",
    ("text", "whitespace_only"): "   ",
    ("text", "very_long"): "x" * 10_000,
    ("timestamp", "epoch"): "1970-01-01T00:00:00Z",
    ("timestamp", "far_past"): "0001-01-01T00:00:00Z",
    ("timestamp", "far_future"): "9999-12-31T23:59:59Z",
    ("boolean", "true"): True,
    ("boolean", "false"): False,
}


def _value_for(boundary_class: str, case: str, declared_type: str | None) -> object:
    if boundary_class == "numeric":
        return _numeric_value(case, declared_type)
    return _STATIC_VALUES[(boundary_class, case)]


class ParameterClassifier:
    """Infers a boundary class for a DSL parameter and proposes edge cases.

    Trains on :data:`TRAINING_EXAMPLES` at construction. Training is
    deterministic (fixed solver and ``random_state``) so a generated suite is
    byte-identical between runs and therefore reviewable in a diff.
    """

    def __init__(self, examples: list[tuple[str, str, str]] | None = None) -> None:
        self._examples = examples if examples is not None else TRAINING_EXAMPLES
        documents = [_document(name, type_) for name, type_, _ in self._examples]
        labels = [label for _, _, label in self._examples]

        self._pipeline = Pipeline(
            [
                # Tokens are already lowercased sub-words, so the analyzer just
                # splits on whitespace; `token_pattern` would otherwise discard
                # the `type=int` marker at the '='.
                (
                    "tfidf",
                    TfidfVectorizer(analyzer=str.split, sublinear_tf=True),
                ),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=0),
                ),
            ]
        )
        self._pipeline.fit(documents, labels)

    @property
    def classes(self) -> list[str]:
        """The boundary classes this model can predict.

        Coerced to plain ``str``: sklearn hands back ``numpy.str_``, which
        compares equal but serialises oddly and would drag numpy types through
        every downstream test case and exported table.
        """
        return [str(label) for label in self._pipeline.named_steps["model"].classes_]

    def classify(self, parameter: Parameter) -> tuple[str, float]:
        """Return the inferred boundary class and the model's confidence.

        Args:
            parameter: A parameter as declared in the diagram.

        Returns:
            ``(class_name, confidence)`` where confidence is the predicted
            probability of the chosen class, in ``(0, 1]``.
        """
        document = _document(parameter.name, parameter.type)
        probabilities = self._pipeline.predict_proba([document])[0]
        best = int(probabilities.argmax())
        return self.classes[best], float(probabilities[best])

    def propose(self, parameter: Parameter) -> BoundaryProposal:
        """Infer a class for ``parameter`` and build its boundary candidates."""
        inferred, confidence = self.classify(parameter)
        candidates = tuple(
            BoundaryCandidate(case=case, value=_value_for(inferred, case, parameter.type))
            for case in BOUNDARY_CASES[inferred]
        )
        return BoundaryProposal(
            parameter=parameter,
            inferred_class=inferred,
            confidence=confidence,
            candidates=candidates,
        )

    def explain(self, parameter: Parameter, top: int = 3) -> list[tuple[str, float]]:
        """Return the tokens that most pushed ``parameter`` toward its class.

        Used by the dashboard to justify an inference. This is why a linear
        model was chosen: the explanation is read straight off the coefficients
        rather than approximated.
        """
        inferred, _ = self.classify(parameter)
        vectorizer = self._pipeline.named_steps["tfidf"]
        model = self._pipeline.named_steps["model"]

        class_index = self.classes.index(inferred)
        coefficients = model.coef_[class_index]
        vocabulary = vectorizer.vocabulary_

        contributions = [
            (token, float(coefficients[vocabulary[token]]))
            for token in _document(parameter.name, parameter.type).split()
            if token in vocabulary
        ]
        contributions.sort(key=lambda item: item[1], reverse=True)
        return contributions[:top]


@lru_cache(maxsize=1)
def get_classifier() -> ParameterClassifier:
    """Return the shared classifier, training it on first use.

    Cached because fitting costs far more than predicting, and every parameter
    in every path would otherwise retrain it.
    """
    return ParameterClassifier()
