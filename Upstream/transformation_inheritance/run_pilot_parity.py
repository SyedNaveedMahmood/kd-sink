# -*- coding: utf-8 -*-
"""run_pilot_parity.py — produce the parity report the E6A pilot gate's criterion 4 needs.

Design §8.5's fourth pilot criterion is "manual vs NNsight attention agree within the Neo
tolerance on 5 examples". ``check_pilot_gate.py`` reads that verdict from a
``parity_report.json`` written by the **frozen** ``run_parity_check`` rather than
recomputing it — a second implementation of the comparison would be exactly the drift the
gate exists to detect.

Nothing could produce that file for a trained student, which is why this script exists.
Three obstacles, all mechanical:

1. **A checkpoint directory contains no tokenizer** (CLAUDE.md trap 7). ``save_pretrained``
   on a model writes weights and config only, and the frozen Neo harness's
   ``--verify-parity`` path does ``AutoTokenizer.from_pretrained(args.model_name)`` on the
   same path it loads the model from. Pointed at ``step_2000/`` it dies before reaching the
   parity check.
2. The harness is **frozen**, so it cannot be given a separate ``--tokenizer`` argument.
3. **The frozen CLI runs three examples, and the criterion asks for five.** It calls
   ``run_parity_check`` without ``sentences=``, which falls back to ``PARITY_SENTENCES`` —
   three sentences spanning the paper's three domains. The report then says
   ``n_sentences: 3`` while its ``rows`` list holds ten *interventions*, which reads like a
   larger sample than it is; the first pilot's criterion 4 passed on it (CLAUDE.md trap 25).

So this script *stages* the checkpoint — it copies the model files and the run's own
tokenizer into one directory that satisfies both loads, never modifying the checkpoint — and
then calls the same ``run_parity_check`` the frozen CLI calls, with
:data:`PILOT_PARITY_SENTENCES` and with every operand assembled by the **frozen module's own
functions**. Nothing about the comparison is reimplemented here: the manual runner, the swap
directions, the massive coordinates, the band and the report are all the frozen code's.

**This is not a scientific measurement.** It is an instrument check: the band the frozen
harness picks (legacy ``compute_band``) only scopes which layers are compared between two
implementations of the *same* forward pass, so trap 1 does not apply. No teacher/student
sink strength is computed here and no number from this file enters a results table.

Usage::

    python transformation_inheritance/run_pilot_parity.py \\
        --run-dir transformation_inheritance/results/e6a/D0/seed0 --step 2000

then pass the printed path to ``check_pilot_gate.py --parity-report``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_NEO_DIR = _REPO / "cross_scale_and_architecture" / "neo"
for _path in (_REPO, _REPO / "common", _REPO / "transformation_inheritance", _NEO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:                                                    # dual-import idiom (CLAUDE.md)
    from . import evaluate_transformation as ev
except ImportError:                                     # pragma: no cover - script use
    import evaluate_transformation as ev

import provenance as prov  # noqa: E402

#: The frozen driver. Never edited; only imported and invoked.
NEO_HARNESS = _NEO_DIR / "intervention_analysis_neo.py"

#: `run_parity_check` writes this name into its output directory.
REPORT_NAME = "parity_report.json"

#: Design §8.5 criterion 4 asks for **five** examples; the frozen ``PARITY_SENTENCES`` default
#: is three. The frozen three are kept as a prefix -- they span the paper's three domains
#: (natural language, code, math), so the pilot's sample stays a superset of what every E1-E5
#: parity report used -- and two are added to reach the registered count. Fixed strings, never
#: drawn from a dataset, exactly as ``train_sentiment_adaptation.PARITY_SENTENCES`` is.
PILOT_PARITY_SENTENCES: Tuple[str, ...] = (
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    "def solve(n):\n    total = 0\n    for i in range(n):\n        total += i * i\n    return total",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May.",
    "Once upon a time there was a little girl named Lily who found a shiny red ball in "
    "the garden.",
    "The chemical symbol for gold is Au, and its atomic number is 79.",
)


def stage_checkpoint(ckpt: Path, tokenizer_source: str, staging: Path, *,
                     tokenizer_revision: Optional[str] = None) -> Path:
    """Copy a checkpoint plus a tokenizer into one directory the frozen CLI can load.

    Returns the staging directory. The checkpoint itself is left untouched — writing the
    tokenizer into it would change a checkpoint's digest after the fact, and
    ``checkpoint_sha256.txt`` is what makes a resumed run auditable.
    """
    from transformers import AutoTokenizer

    source = ev.fingerprint_dir(ckpt)          # merged/ for LoRA, the step root otherwise
    if not (source / "config.json").exists():
        raise SystemExit(f"no model config under {source} — is this a checkpoint?")

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, staging / item.name)

    AutoTokenizer.from_pretrained(
        tokenizer_source, revision=tokenizer_revision).save_pretrained(str(staging))
    if not (staging / "tokenizer.json").exists() and not (
            staging / "vocab.json").exists():
        raise SystemExit(
            f"tokenizer from {tokenizer_source!r} saved no vocabulary into {staging}; the "
            "frozen harness would fall through to an unresolvable HF id")
    return staging


def checkpoint_arch(model_dir: Path) -> str:
    """The architecture key for a staged checkpoint, read from its own ``config.json``.

    From the filesystem, never from the condition name or the run id — the same discipline
    ``evaluate_transformation.fingerprint_dir`` follows, and for the same reason: a rename
    must not be able to point the parity check at the wrong harness.
    """
    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    for name in architectures:
        arch = ev.ARCH_BY_CLASS.get(str(name))
        if arch is not None:
            return arch
    raise SystemExit(
        f"{model_dir}/config.json declares architectures {architectures}, none of which "
        f"maps to a known arch ({sorted(set(ev.ARCH_BY_CLASS.values()))}).")


def _parity_operands(arch: str, model, tokenizer):
    """``(manual_runner_factory, swap_dirs, massive_coords)`` from the **frozen** harness.

    Each entry mirrors that harness's own ``--verify-parity`` branch line for line, so the
    operands cannot drift from what E1–E5 used. Note the two harnesses genuinely differ:
    GPT-Neo's massive-coordinate helper takes the model alone, GPT-2's takes the model and
    the tokenizer. Importing is the sanctioned alternative to editing a frozen file
    (CLAUDE.md rule 1); both modules put their CLI behind an ``if __name__`` guard.
    """
    if arch == "neo":
        import intervention_analysis_neo as neo

        num_heads = (getattr(model.config, "num_heads", None)
                     or model.config.num_attention_heads)
        return (lambda mc: neo.make_manual_runner_neo(model, num_heads, mc),
                neo.neo_swap_directions(model),
                neo.identify_massive_coords_neo(model))
    if arch == "gpt2":
        import importlib
        import intervention_analysis_legacy as gpt2

        # The evaluator's config wrapper patches this module in place; restore the
        # frozen GPT-2 implementation before constructing the parity harness.
        gpt2 = importlib.reload(gpt2)
        return (lambda mc: gpt2.make_manual_runner(model, mc),
                gpt2.gpt2_swap_directions(model),
                gpt2.identify_massive_coords(model, tokenizer))
    raise SystemExit(
        f"no frozen parity harness is wired for architecture {arch!r}. The E6A arms are "
        "neo (TinyStories) and gpt2; add the arch's own frozen operands here rather than "
        "guessing which harness applies.")


def run_frozen_parity(model_dir: Path, out_dir: Path, *, dtype: str = "float32",
                      sentences: Sequence[str] = PILOT_PARITY_SENTENCES,
                      layer_mode: str = "scaled") -> Dict[str, Any]:
    """Run the frozen ``run_parity_check`` on ``model_dir``, on ``sentences``.

    Everything below mirrors the frozen CLI's own ``--verify-parity`` branch: the same
    ``ARCH_SPECS`` entry, the same ``load_nnsight_model``, the same
    ``compute_band(num_layers, layer_mode)``, and the same three operand builders taken
    from the frozen module (see :func:`_parity_operands`) rather than re-derived. The only
    difference is ``sentences``, which no frozen CLI has a flag for and which is the whole
    reason this runs in-process: a subprocess would silently give three examples for a
    criterion that registers five (CLAUDE.md trap 25).

    The architecture comes from the checkpoint, so the same command serves the TinyStories
    arm (GPT-Neo students) and the gpt2 arm (GPT-2 students).

    fp32 only, because ``run_parity_check`` downgrades its gate to advisory in half
    precision: there the manual path and HF are genuinely different algorithms (both
    harnesses upcast q/k to fp32 inside attention) and a mismatch would not be evidence of
    a fault. A pilot gate reading an advisory verdict as a pass would be vacuous.
    """
    import torch
    from transformers import AutoTokenizer

    from intervention_analysis import compute_band
    from nnsight_engine import ARCH_SPECS, NNsightEngine, load_nnsight_model, \
        run_parity_check

    arch = checkpoint_arch(model_dir)
    torch_dtype = {"float32": torch.float32}[dtype]
    spec = ARCH_SPECS[arch]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    lm = load_nnsight_model(spec, str(model_dir), dtype=torch_dtype, tokenizer=tokenizer)
    model = lm._model
    engine = NNsightEngine(lm, spec)

    num_layers = len(model.transformer.h)
    band = compute_band(num_layers, layer_mode)
    manual_runner_factory, swap_dirs, massive_coords = _parity_operands(
        arch, model, tokenizer)
    print(f"[parity] arch {arch}; layers {num_layers}; frozen mid-band "
          f"[{band[0]}, {band[1]}) (layer-mode {layer_mode}); {len(sentences)} examples",
          flush=True)

    return run_parity_check(
        engine, model, tokenizer, band, num_layers,
        manual_runner_factory=manual_runner_factory,
        swap_dirs=swap_dirs,
        massive_coords=massive_coords,
        dtype=dtype, output_dir=str(out_dir), sentences=list(sentences),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="an E6A run directory, e.g. results/e6a/D0/seed0")
    parser.add_argument("--step", type=int, default=2000,
                        help="checkpoint step to check (default: the 2000-step pilot)")
    parser.add_argument("--dtype", default="float32", choices=["float32"],
                        help="fp32 only: the parity gate is advisory in half precision")
    parser.add_argument("--tokenizer", default=None,
                        help="override the tokenizer source (default: from run_config)")
    parser.add_argument("--out", default=None,
                        help="output directory (default: <run-dir>/parity/step_<n>)")
    parser.add_argument("--keep-staging", action="store_true",
                        help="keep the staged checkpoint+tokenizer copy for inspection")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run = ev.read_run_info(run_dir)
    ckpt = run_dir / "checkpoints" / f"step_{args.step}"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}. Train the pilot first, or pass --step.")

    tokenizer_source = ev.resolve_tokenizer_source(
        run, ckpt, smoke=getattr(run, "smoke", False), override=args.tokenizer)
    out_dir = Path(args.out) if args.out else run_dir / "parity" / f"step_{args.step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "_staged_model"

    print(f"[parity] checkpoint      {ckpt}")
    print(f"[parity] tokenizer from  {tokenizer_source}")
    stage_checkpoint(ckpt, tokenizer_source, staging,
                     tokenizer_revision=run.tokenizer_revision)

    report_path = out_dir / REPORT_NAME
    try:
        run_frozen_parity(staging, out_dir, dtype=args.dtype)
    except AssertionError as exc:
        # `run_parity_check` writes the report and *then* raises on a failing row. A failing
        # parity is a result, not a crash: the report is the evidence and the gate reads the
        # verdict out of it. Only a missing report is a failure to run the check.
        print(f"[parity] {exc}", file=sys.stderr)
    if not report_path.exists():
        raise SystemExit(
            f"the frozen parity harness wrote no {REPORT_NAME}. This is a failure to run "
            "the check, not a parity failure; criterion 4 stays unknown rather than false.")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    passed = bool(payload.get("all_rows_pass"))
    n_sentences = payload.get("n_sentences")
    if n_sentences != len(PILOT_PARITY_SENTENCES):
        # The count is the criterion's own wording. If the frozen driver ever stops honouring
        # `sentences=`, say so here rather than letting the gate read a short sample.
        raise SystemExit(
            f"{REPORT_NAME} records n_sentences={n_sentences!r}, expected "
            f"{len(PILOT_PARITY_SENTENCES)}. Design §8.5 criterion 4 registers five "
            "examples; a report on fewer cannot satisfy it.")

    # A sidecar recording what was checked. The report itself is the frozen driver's and is
    # written untouched; provenance goes beside it so nothing overwrites the artefact.
    prov.write_json(out_dir / "parity_context.json", {
        "run_dir": str(run_dir), "checkpoint_step": args.step,
        "checkpoint": str(ckpt), "checkpoint_sha256": ev.checkpoint_sha(ckpt),
        "tokenizer_source": tokenizer_source, "dtype": args.dtype,
        "staged_model_dir": str(staging), "report": str(report_path),
        "all_rows_pass": passed,
        "n_sentences": n_sentences,
        "sentences_source": "run_pilot_parity.PILOT_PARITY_SENTENCES",
        "harness": str(NEO_HARNESS.relative_to(_REPO)).replace("\\", "/"),
        "note": ("Instrument check only: manual vs NNsight agreement on the same forward "
                 "pass. No scientific band or sink measurement is taken here."),
        **prov.provenance_block(),
    })

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\n[parity] all_rows_pass = {passed}")
    print(f"[parity] report  {report_path}")
    print(f"[parity] pass to: check_pilot_gate.py --parity-report {report_path}")
    # Exit 0 whether or not parity passed: the report is the deliverable and the gate reads
    # the verdict from it. Exiting non-zero on a legitimate `false` would make a failing
    # parity indistinguishable from a broken invocation.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
