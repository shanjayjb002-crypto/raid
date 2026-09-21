# RAID — Resilience & Assertion Inference from Diagrams

**Live demo: [softwareengineering.streamlit.app](https://softwareengineering.streamlit.app/)**
— opens empty; press **Load example** to see it populated, or build your own
diagram from scratch.

RAID turns a hand-written architecture diagram into a generated test suite, a
resilience simulation, and a set of chaos experiments — entirely locally, with
no external API calls. Describe your services and call flows in RAID's own
DSL, and RAID will:

- score how analysable the diagram currently is, before generating anything
- enumerate every distinct execution path through a flow (including branches)
- generate functional test cases and boundary-value test cases per path
- run a discrete-event resilience simulation with optional failure injection
- fuzz the flow with random mutations and write up the regressions it finds
- trace every generated artifact back to the diagram element that produced it

All of it is browsable from a Streamlit dashboard with two input modes: a
visual form builder, or a raw DSL text editor.

## Quickstart

Requires Python 3.11+ (developed and tested on 3.13).

```bash
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run raid/dashboard/app.py

# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run raid/dashboard/app.py
```

This opens the dashboard at `http://localhost:8501`. It starts **empty** —
press **Load example** to populate the sample "Login" diagram, or build one
from scratch.

Run the test suite:

```bash
.venv/bin/python -m pytest raid/tests -q
```

277 tests, ~7 seconds, no network access required.

## Example input (RAID DSL)

```
service OrderService
service PaymentService
service InventoryService

flow PlaceOrder:
  OrderService -> InventoryService : checkStock(itemId:int, qty:int)
  alt stock_available:
    OrderService -> PaymentService : charge(amount:float, cardId:string) [timeout=2s, retry=3]
    alt payment_success:
      OrderService -> InventoryService : reserve(itemId:int, qty:int)
  else out_of_stock:
    OrderService -> OrderService : rejectOrder()
```

`service` declares a participant, `flow` declares a named scenario, `->`
declares one directed call with typed parameters, and `alt`/`else` declare a
branch. Annotations like `[timeout=2s, retry=3]` are optional and drive the
resilience simulation.

## Project structure

```
raid/
  dsl/          grammar + parser (lark)              — the only input format
  graph/        graph model built from parsed DSL     — networkx
  testgen/      path enumeration + test generation    — incl. scikit-learn
                boundary-value classifier
  simulate/     discrete-event resilience simulation  — simpy
  fuzz/         mutation engine + chaos test writer
  trace/        cross-concern impact graph            — links diagram
                elements to tests / findings / specs
  sufficiency/  diagram completeness / ambiguity scorer
  dashboard/    Streamlit app tying it all together
  tests/        pytest suite — one file per module above
```

Every module is independently testable with pytest before it's wired into the
dashboard. See [`PROJECT_RULES.md`](PROJECT_RULES.md) for the full set of project rules
(local-only intelligence, no external diagram notations, fixed dependency
stack) that every change in this repo follows.

## Dependencies

| Package | Used for |
| --- | --- |
| `lark-parser` | DSL grammar and parsing |
| `networkx` | Service graph, path enumeration, impact graph |
| `scikit-learn` | Boundary-value classifier |
| `simpy` | Discrete-event resilience simulation |
| `deap` | Declared for the mutation engine (currently a custom random-search fuzzer — see below) |
| `streamlit` | Dashboard |
| `pytest` | Test suite |

## Known limitations

- The fuzzer (`raid/fuzz`) currently uses a custom random-mutation strategy
  rather than `deap`. `deap` earns its place once mutations become
  *combinations* of faults rather than single mutations — see the module
  docstring in `raid/fuzz/mutations.py` for the full reasoning.
- Retry annotations (`[retry=...]`) are parsed and scored, but not yet acted
  on by the simulation — a timed-out call currently fails once.
- The visual dashboard builder supports one level of `alt`/`else` nesting;
  deeper nesting (as in the example above) needs the "Edit as code" tab.
- `lark-parser` 0.12.0 is the legacy PyPI package name (last released 2021).
  It works but emits deprecation warnings on Python 3.13; migrating to the
  maintained `lark` 1.x package is a small, worthwhile follow-up.

## Deployment

The dashboard is a Streamlit app and needs a persistent Python process with
WebSocket support — it is not deployable to static/serverless hosts like
Vercel. It deploys cleanly to
[Streamlit Community Cloud](https://streamlit.io/cloud) (free): point it at
this repo with the main file path `raid/dashboard/app.py`.
