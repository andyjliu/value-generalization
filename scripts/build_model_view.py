"""Build a revision-pinned *view* of an HF checkpoint under
``<finetune_root>/model_views/<repo short name>``, the ``--model`` path every
paper predictor was built from.

Two kinds:

- default: a directory of symlinks to one snapshot's files plus a rewritten
  ``tokenizer_config.json`` the pinned ``ws-train`` env (transformers 4.x)
  can load. Checkpoints saved by transformers 5.x write
  ``extra_special_tokens`` as a list, which 4.x reads as a dict and dies on;
  those 5.x-only keys are dropped, and the chat template is inlined from
  ``chat_template.jinja`` (with render-neutral ``{% generation %}`` markers
  stripped) so 4.x sees the same bytes 5.x does. Used for the neutral-SFT
  checkpoints.
- ``--symlink``: the view is a plain symlink to the snapshot. Used for the
  pretrained base models, which load in every env unchanged.

The snapshot is downloaded at ``--revision`` if it is not in ``$HF_HOME`` yet,
and never modified. Naming the view after the HF repo's short name keeps every
store path (``model_short``) identical to what the HF id would give.

Note: data and predictor artifact IDs hash the ``--model`` path, so they match
the paper's IDs only when ``finetune_root`` is the same path the paper used;
elsewhere the artifacts get new IDs with identical contents-by-recipe.

    .venvs/core/bin/python scripts/build_model_view.py value-generalization/neutral-sft-v3-qwen3-8b \
        --revision f1550fe714e01e2116cc4fdf3a1d510f8fd379c7
    .venvs/core/bin/python scripts/build_model_view.py Qwen/Qwen3-8B-Base \
        --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --symlink
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from valuegen.config import load_cluster

DROP_KEYS = ("extra_special_tokens", "is_local", "local_files_only", "backend")

# transformers/TRL register a `{% generation %}...{% endgeneration %}` Jinja
# extension that marks the assistant span for `assistant_only_loss` (used only
# by apply_chat_template(return_assistant_tokens_mask=True)). The tags EMIT
# NOTHING -- rendered output is byte-identical with or without them. But
# axolotl (the ws-train LoRA trainer) parses the template with a *plain*
# jinja2.Environment that has no such extension registered, so it dies with
# "Encountered unknown tag 'generation'" before any training starts (OLMo-3's
# shipped template carries these markers). axolotl 0.8.0 builds its
# assistant-loss mask from `roles_to_train`, not from these markers, so
# stripping them is safe and lets `chat_template: tokenizer_default` load the
# view unchanged. Rendered bytes are unaffected (verified against the un-
# stripped template), so persona/grad extraction through the same view is
# identical. Only the block tags are removed; `add_generation_prompt` (a
# variable, not a tag) is left alone.
_GENERATION_TAGS = (
    "{%- generation -%}", "{% generation %}",
    "{%- endgeneration -%}", "{% endgeneration %}",
)


def strip_generation_markers(template: str) -> tuple[str, bool]:
    """Remove render-neutral `{% generation %}` block tags. Returns (template,
    stripped?)."""
    out = template
    for tag in _GENERATION_TAGS:
        out = out.replace(tag, "")
    return out, (out != template)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--revision", required=True, help="snapshot sha (full)")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--cluster", default=None)
    ap.add_argument("--symlink", action="store_true",
                    help="make the view a plain symlink to the snapshot (base models)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cluster = load_cluster(args.cluster)
    view = cluster.finetune_root / "model_views" / args.repo_id.split("/")[-1]
    # An existing view is left alone (checked before touching the HF cache, so
    # a re-run never downloads): re-runs are no-ops, and a regenerate script
    # can call this unconditionally.
    if view.is_symlink() or view.exists():
        if not ((view / "VIEW.json").is_file() or view.is_symlink()):
            if not args.force:
                raise SystemExit(f"{view} exists and is not a view; --force to replace")
        elif not args.force:
            print(f"{view} exists (pass --force to rebuild)")
            return
        if view.is_symlink():
            view.unlink()
    if not args.hf_home:
        raise SystemExit("--hf-home or $HF_HOME required")
    snap = (Path(args.hf_home) / "hub" / f"models--{args.repo_id.replace('/', '--')}"
            / "snapshots" / args.revision)
    if not snap.is_dir():
        from huggingface_hub import snapshot_download

        snapshot_download(args.repo_id, revision=args.revision,
                          cache_dir=Path(args.hf_home) / "hub")
    view.parent.mkdir(parents=True, exist_ok=True)
    if args.symlink:
        if view.is_dir():
            raise SystemExit(f"{view} is a directory view; remove it to make a symlink")
        view.symlink_to(snap.resolve())
        print(view)
        return
    view.mkdir(parents=True, exist_ok=True)
    for f in view.iterdir():
        f.unlink()

    # chat_template.jinja is written (not symlinked) below with markers stripped:
    # transformers prefers a chat_template.jinja *file* over the inlined
    # tokenizer_config["chat_template"], so a symlink to the original
    # marker-bearing file would defeat the strip and reach axolotl's parser.
    for src in sorted(snap.iterdir()):
        if src.name in ("tokenizer_config.json", "chat_template.jinja"):
            continue
        (view / src.name).symlink_to(src.resolve())
    tok_cfg = json.loads((snap / "tokenizer_config.json").read_text())
    dropped = [k for k in DROP_KEYS if k in tok_cfg]
    for k in dropped:
        tok_cfg.pop(k)
    # transformers 5.x writes tokenizer_class="TokenizersBackend" (a generic
    # fast backend) that 4.x cannot resolve ("Tokenizer class TokenizersBackend
    # does not exist"). Drop it so AutoTokenizer falls back to the fast tokenizer
    # in tokenizer.json. Only the 5.x sentinel is dropped -- a real 4.x class
    # (Qwen2Tokenizer, etc.) is left intact.
    if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
        tok_cfg.pop("tokenizer_class")
        dropped.append("tokenizer_class")
    jinja = snap / "chat_template.jinja"
    markers_stripped = False
    if jinja.is_file():
        tmpl, markers_stripped = strip_generation_markers(jinja.read_text())
        tok_cfg["chat_template"] = tmpl
        # real file (not a symlink to the marker-bearing original), so every
        # loader -- transformers (prefers the file) and the inlined config --
        # sees the same marker-free template.
        (view / "chat_template.jinja").write_text(tmpl)
    (view / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n")
    (view / "VIEW.json").write_text(json.dumps({
        "repo_id": args.repo_id, "revision": args.revision, "snapshot": str(snap),
        "dropped_tokenizer_config_keys": dropped,
        "chat_template_inlined_from": "chat_template.jinja" if jinja.is_file() else None,
        # render-neutral marker strip so axolotl's plain-jinja parser can read it
        "generation_markers_stripped": markers_stripped,
    }, indent=2) + "\n")
    print(view)


if __name__ == "__main__":
    main()
