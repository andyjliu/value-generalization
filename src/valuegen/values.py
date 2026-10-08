"""Value-set registry.

Value sets live in ``value_sets/*.json`` at the repo root — the canonical
registry. :func:`load_value_set` **raises** on a miss, unlike conflictscope's
``utils.load_value_dict``, which silently falls back to an unrelated 5-value
dict (see the ``_external.py`` docstring).
"""

from __future__ import annotations

import json
from pathlib import Path

from valuegen._external import value_sets_dir

# ── Value-set registry ───────────────────────────────────────────────────────


def list_value_sets() -> list[str]:
    """Names of every value set shipped in the repo registry."""
    return sorted(p.stem for p in value_sets_dir().glob("*.json"))


def load_value_set(name_or_path: str | Path) -> dict[str, str]:
    """Load a value set as ``{value_name: description}``.

    ``name_or_path`` is either a bare registry name (``"constitution_tenets_v3"``) or a
    path to a JSON file. Raises ``FileNotFoundError`` on a miss — never falls
    back.
    """
    path = Path(name_or_path)
    if not path.suffix == ".json":
        path = value_sets_dir() / f"{path}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No value set at {path}. Known sets: {', '.join(list_value_sets())}"
        )
    with open(path) as f:
        return json.load(f)


def registry_name(name_or_path: str | Path) -> str:
    """Normalize a value-set reference to its **registry name**.

    Data configs identify a value set by name, not path: the name is both a
    filesystem component (``data/elicitation/.../{value_set}/...``) and a config-
    hash input, so it has to be short and machine-independent. A path is
    therefore only accepted when it points *into* the registry, and is reduced
    to its stem; a path anywhere else raises.

    Silently keeping the stem of an external path is the bug this exists to
    prevent: ``/tmp/HHH.json`` would resolve back to the registry's unrelated
    ``HHH``, building the wrong value set under an identical config hash.
    """
    path = Path(name_or_path)
    if path.suffix != ".json":
        name = str(name_or_path)
        if not (value_sets_dir() / f"{name}.json").is_file():
            raise FileNotFoundError(
                f"No value set named {name!r}. Known sets: {', '.join(list_value_sets())}"
            )
        return name

    resolved = path.resolve()
    if resolved.parent != value_sets_dir().resolve():
        raise ValueError(
            f"Value set {resolved} is outside the registry ({value_sets_dir()}). "
            "Configs identify value sets by name, so external files cannot be "
            f"referenced by path; install it first:\n"
            f"    cp {resolved} {value_sets_dir() / resolved.name}\n"
            f"then pass --value-set {resolved.stem}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"No value set at {resolved}. Known sets: {', '.join(list_value_sets())}"
        )
    return resolved.stem


def value_set_cli_arg(name_or_path: str | Path) -> str:
    """Value-set argument safe to pass to conflictscope subprocess CLIs.

    conflictscope's ``generate_scenarios.py``/``filter_scenarios.py`` resolve
    ``--value-set`` through ``utils.load_value_dict(name)``, which joins
    ``{submodule}/value_sets/{name}.json`` — and on a miss silently returns an
    unrelated 5-value fallback. But ``os.path.join`` discards
    the preceding components when handed an absolute path, so an **absolute
    path minus the ``.json`` suffix** resolves to exactly our registry file
    through the existing code, no submodule change needed. This helper builds
    that argument and raises here if the file doesn't exist, so the silent
    fallback is unreachable.
    """
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = value_sets_dir() / f"{path}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No value set at {path}. Known sets: {', '.join(list_value_sets())}"
        )
    return str(path.resolve())[: -len(".json")]
