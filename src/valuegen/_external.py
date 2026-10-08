"""Single place where the vendored submodules are reached.

Only **one** submodule ever goes on ``sys.path``: ``external/conflictscope/src``.
Its modules import each other flat (``import model_wrappers``, ``import utils``),
so its ``src/`` has to be importable as a top-level directory. Everything that
needs those modules calls :func:`load_conflictscope` rather than doing its own
path surgery::

    from valuegen._external import load_conflictscope
    cs = load_conflictscope()
    wrapper = cs.model_wrappers.ModelWrapper.create("gpt-4.1")

``load_conflictscope`` is idempotent and safe to call from any entry point.

persona_vectors and weight-steering are **not** importable this way, by design.
Both are script-oriented (their orchestrators shell out: ``python generate_vec.py``,
``python -m eval.eval_persona``) and both expect their own conda env — the persona
env has ``unsloth``, ``conflictbench`` does not. Putting them on ``sys.path``
alongside conflictscope also collides: ``external/persona_vectors/utils.py`` and
``external/conflictscope/src/utils.py`` are different modules, both imported as a
bare ``import utils``, and whichever loads first wins ``sys.modules['utils']`` for
the whole process. Use :func:`persona_vectors_root` / :func:`weight_steering_root`
to locate them and run them as subprocesses with the right env.

Note on lazy imports: ``utils`` is stdlib-only and cheap, so it is imported
eagerly. ``model_wrappers`` is **not** cheap — it does a module-level
``import torch`` and ``from vllm import LLM`` — so it is lazy along with the rest.
Callers that only build matrices or run analysis never pay for the GPU stack.
Reaching ``cs.model_wrappers`` requires the ``[vllm]`` extra.

Note on value sets: conflictscope's ``utils.load_value_dict(name)`` resolves
``name`` against the *submodule's* ``value_sets/``, which only ships the three sets
that repo was published with. On a miss it does **not** raise — it prints an error
and returns the unrelated 5-value ``utils.VALUE_DICT`` fallback. Pass value
dictionaries explicitly across the boundary, or resolve paths with
:func:`value_sets_dir`; never let a bare name reach the submodule.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = REPO_ROOT / "external"

CONFLICTSCOPE_SRC = EXTERNAL / "conflictscope" / "src"
PERSONA_VECTORS = EXTERNAL / "persona_vectors"
WEIGHT_STEERING = EXTERNAL / "weight-steering"
MODEL_SPEC_MIDTRAINING = EXTERNAL / "model_spec_midtraining"


class SubmoduleMissingError(RuntimeError):
    """Raised when a submodule directory is absent or was never checked out."""


class ModuleShadowError(RuntimeError):
    """Raised when a foreign module has already claimed a conflictscope name."""


def value_sets_dir() -> Path:
    """The canonical value-set registry (valuegen's, not any submodule's)."""
    return REPO_ROOT / "value_sets"


def _require(path: Path, name: str) -> Path:
    if not path.is_dir() or not any(path.iterdir()):
        raise SubmoduleMissingError(
            f"external submodule {name!r} is missing or empty at {path}.\n"
            f"Run: git submodule update --init --recursive"
        )
    return path


def persona_vectors_root() -> Path:
    """Filesystem root of the persona_vectors fork. Run it as a subprocess."""
    return _require(PERSONA_VECTORS, "external/persona_vectors")


def weight_steering_root() -> Path:
    """Filesystem root of the weight-steering fork. Run it as a subprocess."""
    return _require(WEIGHT_STEERING, "external/weight-steering")


def model_spec_midtraining_root() -> Path:
    """Filesystem root of the model_spec_midtraining fork.

    Run it as a subprocess with ``PYTHONPATH`` set to this root — never
    ``pip install`` it: it packages itself as a literal top-level ``src``
    package, which would shadow any other flat-imported module tree.
    """
    return _require(MODEL_SPEC_MIDTRAINING, "external/model_spec_midtraining")


def _check_not_shadowed(src: Path) -> None:
    """Fail loudly if something else already owns a name conflictscope needs.

    conflictscope's modules resolve each other by bare top-level name, so a
    same-named module imported earlier (persona_vectors' ``utils`` is the live
    example) would be handed to them instead, silently.
    """
    for name in ("utils", "model_wrappers"):
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        if Path(origin).resolve().parent != src:
            raise ModuleShadowError(
                f"sys.modules[{name!r}] is already bound to {origin}, not "
                f"conflictscope's copy in {src}. Something else was put on "
                f"sys.path; conflictscope must be the only submodule there. Run "
                f"the other submodule as a subprocess instead."
            )


def load_conflictscope() -> types.SimpleNamespace:
    """Import the conflictscope core modules and return them as one namespace.

    Only ``utils`` is imported eagerly (stdlib-only). Every other module is
    exposed as a lazily-imported attribute; ``model_wrappers`` and
    ``evaluate_models`` pull in torch/vllm and need the ``[vllm]`` extra.
    """
    src = _require(CONFLICTSCOPE_SRC, "external/conflictscope")

    entry = str(src)
    if entry not in sys.path:
        sys.path.insert(0, entry)

    _check_not_shadowed(src)

    import utils  # noqa: PLC0415  (path must be set first)

    class _Lazy(types.SimpleNamespace):
        _LAZY = (
            "model_wrappers",
            "evaluate_models",
            "simulated_conversation",
            "generate_scenarios",
            "filter_scenarios",
            "analyze_experiment",
        )

        def __getattr__(self, name: str):
            if name not in self._LAZY:
                raise AttributeError(name)
            import importlib

            _check_not_shadowed(src)
            module = importlib.import_module(name)
            setattr(self, name, module)
            return module

    return _Lazy(root=src.parent, src=src, utils=utils)
