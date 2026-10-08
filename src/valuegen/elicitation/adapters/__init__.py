"""Per-fork adapters: invocation + minimal schema conversion.

Each adapter exposes ``build(cfg, cluster, artifact, only_values=None)`` and
does the smallest thing that turns its method's native machinery into the
standard artifact. Method math stays in the forks / submodule.
"""
