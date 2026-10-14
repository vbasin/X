"""src.search — Phase 23 W3: full-fidelity model-selection search controller.

Runs every catalogued model family (and ensembles) to a complete walk-forward
evaluation, scores on the Frozen Evaluation Contract, and selects a champion.

Public API::

    from src.search import run_search
    result = run_search("nq-50pt")
"""

# Lazy: importing controller here would fire RuntimeWarning under
# ``python -m src.search.controller`` (package import loads the module, then
# runpy executes it again as __main__).
def __getattr__(name: str):
    if name == "run_search":
        from .controller import run_search
        return run_search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_search"]
