# -*- coding: utf-8 -*-
"""test_pilot_parity_staging.py — the pilot parity report can actually be produced.

Design §8.5 criterion 4 reads a ``parity_report.json`` written by the frozen
``run_parity_check``. Nothing could produce one for a trained student, for a reason that is
purely mechanical and entirely invisible until you try it: ``save_pretrained`` on a model
writes weights and config only, so a checkpoint directory has **no tokenizer**
(CLAUDE.md trap 7), while the frozen Neo harness loads model *and* tokenizer from the same
``--model-name`` path and cannot be given a second one.

``run_pilot_parity.stage_checkpoint`` closes that gap by copying both into one directory.
These tests assert the three properties that make the staging trustworthy rather than
merely convenient:

* the staged directory satisfies both loads;
* the checkpoint is left **unmodified** — writing a tokenizer into it would change a
  checkpoint's contents after ``checkpoint_sha256.txt`` was computed;
* a LoRA checkpoint stages its ``merged/`` copy, never the PEFT adapter.

The parity *numbers* are not tested here — they are the frozen harness's business, and a
test asserting them would be a second implementation of the comparison the gate exists to
detect.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_pilot_parity as rpp  # noqa: E402
import train_distillation as td  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    """The offline word-level tokenizer a smoke run writes, built the same way."""
    root = tmp_path_factory.mktemp("tok")
    td.build_smoke_assets(root)
    return root / "tokenizer"


def _fake_checkpoint(root: Path, *, merged: bool = False) -> Path:
    """A checkpoint directory with model files and — as in production — no tokenizer."""
    ckpt = root / "checkpoints" / "step_2000"
    target = (ckpt / "merged") if merged else ckpt
    target.mkdir(parents=True)
    (target / "config.json").write_text(json.dumps({"model_type": "gpt_neo"}),
                                        encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"\x00weights\x00")
    if merged:                      # the adapter the evaluator must never fingerprint
        (ckpt / "adapter").mkdir()
        (ckpt / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    return ckpt


def test_staging_produces_a_directory_that_satisfies_both_loads(tmp_path, tokenizer_dir):
    from transformers import AutoTokenizer

    ckpt = _fake_checkpoint(tmp_path)
    staged = rpp.stage_checkpoint(ckpt, str(tokenizer_dir), tmp_path / "staged")

    assert (staged / "config.json").exists(), "the model load needs a config"
    assert (staged / "model.safetensors").exists(), "the model load needs weights"
    # The load that used to fail outright.
    assert AutoTokenizer.from_pretrained(str(staged)) is not None


def test_the_checkpoint_itself_is_not_modified(tmp_path, tokenizer_dir):
    """Staging copies; it must not write a tokenizer into the checkpoint.

    ``checkpoint_sha256.txt`` is what makes a resumed run auditable, so adding files to a
    checkpoint after the fact would silently invalidate it.
    """
    ckpt = _fake_checkpoint(tmp_path)
    before = sorted(p.name for p in ckpt.rglob("*"))

    rpp.stage_checkpoint(ckpt, str(tokenizer_dir), tmp_path / "staged")

    assert sorted(p.name for p in ckpt.rglob("*")) == before
    assert not (ckpt / "tokenizer.json").exists()


def test_a_lora_checkpoint_stages_the_merged_copy_not_the_adapter(tmp_path, tokenizer_dir):
    """`03` §4.5: only the merged copy is ever loaded by the frozen harness.

    Resolved from the weights on disk via ``fingerprint_dir``, not from a condition name,
    so a rename cannot point the parity check at a PEFT adapter.
    """
    ckpt = _fake_checkpoint(tmp_path, merged=True)
    staged = rpp.stage_checkpoint(ckpt, str(tokenizer_dir), tmp_path / "staged")

    assert (staged / "config.json").exists()
    assert not (staged / "adapter_config.json").exists(), (
        "the PEFT adapter was staged; the frozen harness cannot resolve module paths "
        "through a PEFT wrapper")


def test_a_directory_without_weights_is_refused(tmp_path, tokenizer_dir):
    """An empty or wrong path must fail loudly, not stage something unloadable."""
    empty = tmp_path / "checkpoints" / "step_2000"
    empty.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no model config"):
        rpp.stage_checkpoint(empty, str(tokenizer_dir), tmp_path / "staged")


def test_staging_is_idempotent(tmp_path, tokenizer_dir):
    """Re-running after an interruption must not merge two states (CLAUDE.md: resumable)."""
    ckpt = _fake_checkpoint(tmp_path)
    staged = rpp.stage_checkpoint(ckpt, str(tokenizer_dir), tmp_path / "staged")
    (staged / "stale_leftover.bin").write_bytes(b"junk")

    staged = rpp.stage_checkpoint(ckpt, str(tokenizer_dir), tmp_path / "staged")
    assert not (staged / "stale_leftover.bin").exists()


def test_the_frozen_harness_is_invoked_not_reimplemented():
    """The script must dispatch to the frozen driver (CLAUDE.md rule 3).

    Asserted on the source because the alternative — a second parity implementation — is
    exactly the drift design §8.5 criterion 4 exists to detect, and it would still pass a
    behavioural test. The script no longer shells out to the frozen CLI (it cannot be given
    ``sentences=``), so what is pinned here is stronger: every operand still comes from the
    frozen module's own functions, and the comparison itself is still ``run_parity_check``.
    """
    source = (REPO / "transformation_inheritance"
              / "run_pilot_parity.py").read_text(encoding="utf-8")
    assert "import intervention_analysis_neo as neo" in source
    for frozen_operand in ("neo.make_manual_runner_neo(", "neo.neo_swap_directions(",
                           "neo.identify_massive_coords_neo(", "run_parity_check("):
        assert frozen_operand in source, (
            f"{frozen_operand!r} is gone; the operands or the comparison are no longer the "
            "frozen harness's, which is the drift criterion 4 exists to detect")
    for reimplemented in ("def verify_parity", "def run_parity_check",
                          "max_abs_metric_deviation ="):
        assert reimplemented not in source, (
            f"{reimplemented!r} suggests the parity comparison was reimplemented rather "
            "than dispatched to the frozen harness")


def test_the_frozen_operands_exist_on_the_frozen_module():
    """The three operand builders and the driver must still be importable by those names.

    A rename inside the frozen tree would otherwise surface as an ``AttributeError`` in the
    middle of a GPU run rather than here.
    """
    import intervention_analysis_neo as neo  # noqa: WPS433 - importing is the assertion
    from nnsight_engine import run_parity_check  # noqa: F401

    for name in ("make_manual_runner_neo", "neo_swap_directions",
                 "identify_massive_coords_neo"):
        assert callable(getattr(neo, name, None)), f"frozen Neo harness lost {name}"


def test_the_pilot_supplies_the_five_registered_examples():
    """Design §8.5 criterion 4 says five; the frozen default is three.

    ``run_parity_check`` falls back to ``nnsight_engine.PARITY_SENTENCES`` when it is given
    none, and the frozen Neo CLI gives it none — so the first pilot's report said
    ``n_sentences: 3`` while criterion 4 read the ten-row ``rows`` list and passed
    (CLAUDE.md trap 25). The frozen three are kept as a prefix so the pilot's sample is a
    superset of every E1–E5 parity report's, not a different sample.
    """
    from nnsight_engine import PARITY_SENTENCES

    assert len(rpp.PILOT_PARITY_SENTENCES) == 5
    assert len(set(rpp.PILOT_PARITY_SENTENCES)) == 5
    assert all(isinstance(s, str) and s.strip() for s in rpp.PILOT_PARITY_SENTENCES)
    assert rpp.PILOT_PARITY_SENTENCES[:len(PARITY_SENTENCES)] == tuple(PARITY_SENTENCES), (
        "the frozen three-domain sentences must remain the prefix, so the pilot's sample "
        "extends the E1-E5 one instead of replacing it")

    # The default is the registered set: a caller cannot silently get three.
    import inspect

    default = inspect.signature(rpp.run_frozen_parity).parameters["sentences"].default
    assert tuple(default) == rpp.PILOT_PARITY_SENTENCES


def test_the_gate_and_the_producer_agree_on_the_example_count():
    """One number, two files: the count the gate requires and the count this script runs."""
    import check_pilot_gate as cpg

    assert cpg.PARITY_EXAMPLES_REQUIRED == len(rpp.PILOT_PARITY_SENTENCES)


# ── architecture dispatch: the TinyStories arm is GPT-Neo, the gpt2 arm is GPT-2 ─────


def _config_dir(tmp_path, architecture):
    path = tmp_path / architecture
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"architectures": [architecture], "num_layers": 6}), encoding="utf-8")
    return path


def test_the_architecture_comes_from_the_checkpoint_not_the_run_name(tmp_path):
    """From the filesystem, so a rename cannot point parity at the wrong harness."""
    assert rpp.checkpoint_arch(_config_dir(tmp_path, "GPTNeoForCausalLM")) == "neo"
    assert rpp.checkpoint_arch(_config_dir(tmp_path, "GPT2LMHeadModel")) == "gpt2"


def test_an_unknown_architecture_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="maps to a known arch"):
        rpp.checkpoint_arch(_config_dir(tmp_path, "LlamaForCausalLM"))
    with pytest.raises(SystemExit, match="no frozen parity harness"):
        rpp._parity_operands("opt", object(), object())


def test_both_frozen_harnesses_expose_the_operands_the_dispatch_uses():
    """A rename inside the frozen tree must fail here, not mid-run on the GPU.

    The two harnesses genuinely differ — GPT-Neo's massive-coordinate helper takes the
    model alone, GPT-2's takes the model and the tokenizer — so the signatures are asserted
    as well as the names.
    """
    import inspect

    import intervention_analysis_legacy as gpt2
    import intervention_analysis_neo as neo

    for name in ("make_manual_runner_neo", "neo_swap_directions",
                 "identify_massive_coords_neo"):
        assert callable(getattr(neo, name, None)), f"frozen Neo harness lost {name}"
    for name in ("make_manual_runner", "gpt2_swap_directions", "identify_massive_coords"):
        assert callable(getattr(gpt2, name, None)), f"frozen GPT-2 harness lost {name}"

    assert len(inspect.signature(neo.identify_massive_coords_neo).parameters) == 1
    assert len(inspect.signature(gpt2.identify_massive_coords).parameters) >= 2


def test_the_dispatch_mirrors_each_frozen_cli():
    """Asserted on the frozen sources: the operands must be the ones those CLIs pass."""
    neo_src = (REPO / "cross_scale_and_architecture" / "neo"
               / "intervention_analysis_neo.py").read_text(encoding="utf-8")
    gpt2_src = (REPO / "common" / "intervention_analysis_legacy.py").read_text(
        encoding="utf-8")
    ours = (REPO / "transformation_inheritance"
            / "run_pilot_parity.py").read_text(encoding="utf-8")

    for token in ("make_manual_runner_neo(", "neo_swap_directions(",
                  "identify_massive_coords_neo("):
        assert token in neo_src and token in ours, token
    for token in ("make_manual_runner(", "gpt2_swap_directions(",
                  "identify_massive_coords("):
        assert token in gpt2_src and token in ours, token
