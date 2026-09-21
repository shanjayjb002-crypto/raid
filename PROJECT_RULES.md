# RAID — Resilience & Assertion Inference from Diagrams

This file records non-negotiable rules for every future session working on this
project. These rules override default assumptions and must not be relaxed
without the user explicitly changing them here.

## Non-negotiable rules

1. **No external API calls of any kind.** No OpenAI, Anthropic, Gemini, or any
   cloud AI service. All intelligence must be local: rule-based logic,
   scikit-learn, or classical algorithms only.

2. **No use of PlantUML, Mermaid, or any existing diagram notation.** All
   diagram input goes through our own custom DSL, defined in phase 1
   (`raid/dsl/`).

3. **Stack** (do not substitute or add alternatives without discussion):
   - Python 3.11+
   - `lark-parser` — DSL grammar and parsing
   - `networkx` — graph modeling
   - `scikit-learn` — boundary-value classifier
   - `simpy` — resilience simulation
   - `deap` — mutation/fuzzing engine
   - `streamlit` — dashboard

4. **Project structure:**
   ```
   raid/
     dsl/          -> grammar + parser
     graph/        -> graph model built from parsed DSL
     testgen/      -> path enumeration + boundary-value test generation
     simulate/     -> SimPy resilience simulation
     fuzz/         -> mutation engine + chaos test spec writer
     trace/        -> cross-concern impact graph linking diagram elements
                       to tests/resilience findings/threat flags
     sufficiency/  -> diagram completeness/ambiguity scorer
     dashboard/    -> Streamlit app tying it all together
     tests/        -> pytest unit tests for every module above
   ```

5. **Every module must be independently testable with pytest** before it's
   wired into the dashboard. Do not couple a module's core logic to Streamlit
   session state or UI code — keep logic importable and testable in isolation.

6. **Write clear docstrings.** This is a university project and every design
   decision needs to be explainable in a viva. Docstrings should explain *why*
   a design choice was made, not just restate the function signature.

## Notes for future sessions

- The DSL (`raid/dsl/`) is the single source of truth for diagram input. Do
  not add parsers or adapters for external diagram formats.
- Keep classifiers/heuristics in `sufficiency/` and `testgen/` local
  (scikit-learn or rule-based) — never call out to a hosted model.
- Scaffolding only was set up initially (folder structure, empty
  `__init__.py` files, venv, dependencies). No feature code exists yet as of
  project creation.
