# -*- coding: utf-8 -*-
"""test_resume_exactness.py — ``--resume auto`` is bit-exact (WP5).

``06_TEST_PLAN.md`` row 27: train 20 steps, checkpoint at 10, resume, and assert the
step-20 weights are bit-identical to the uninterrupted run.

This is not a convenience feature. E6A is 1.5–3 GPU-days on a single 4080 SUPER; a resume
that is merely *approximately* correct would silently make the second half of every
interrupted run a different experiment from the first, with no error and no way to tell
after the fact which runs were affected.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "transformation_inheritance"))

import provenance as prov  # noqa: E402
import train_distillation as td  # noqa: E402

CHECKPOINTS = (0, 10, 20)


def _config():
    return {
        "condition": "D2",
        "loss": {"ce_weight": 0.45, "kd_weight": 0.45, "attn_weight": 0.10,
                 "temperature": 2.0, "layer_map": td.DEFAULT_LAYER_MAP},
        "data": {"block_size": 32},
        "optim": {"max_steps": 20, "lr": 1e-3, "warmup_steps": 2, "grad_clip": 1.0},
        "checkpoints": list(CHECKPOINTS),
    }


def _run(root: Path, *, max_steps: int, resume: str, stop_after=None):
    setup = td.prepare_run(_config(), seed=0, output_dir=root, max_steps=max_steps,
                           smoke=True)
    # smoke mode forces batch 2 / accum 1 / warmup 1; keep it, but let the run go the
    # full 20 steps rather than the 5-step default.
    setup.max_steps = max_steps
    td.train(setup, resume=resume, stop_after=stop_after)
    return setup


def _weights(run_dir: Path, step: int):
    from safetensors.torch import load_file
    return load_file(str(run_dir / "checkpoints" / f"step_{step}" / "model.safetensors"))


def test_resume_reproduces_the_uninterrupted_weights_bit_for_bit():
    """The canonical check: interrupt after step 10, resume, compare at step 20.

    The interruption is simulated by discarding the final checkpoint of a completed run
    rather than by training a *shorter* run. That distinction matters: the cosine schedule
    is defined over ``max_steps``, so a run configured for 10 steps has a different
    learning rate at every step than one configured for 20 and would not be comparable.
    A real interruption keeps the config and loses only the progress.
    """
    with tempfile.TemporaryDirectory(prefix="e6a_resume_",
                                     ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)

        reference_root = tmp / "reference"
        reference = _run(reference_root, max_steps=20, resume="none")
        reference_final = _weights(reference.run_dir, 20)

        # An interrupted run: same config, progress lost after step 10.
        interrupted_root = tmp / "interrupted"
        shutil.copytree(reference_root, interrupted_root)
        interrupted_run_dir = interrupted_root / "e6a" / "D2" / "seed0"
        shutil.rmtree(interrupted_run_dir / "checkpoints" / "step_20")
        assert td.latest_checkpoint(interrupted_run_dir).name == "step_10"

        resumed = _run(interrupted_root, max_steps=20, resume="auto")
        resumed_final = _weights(resumed.run_dir, 20)

        assert set(resumed_final) == set(reference_final)
        for name in reference_final:
            assert torch.equal(reference_final[name], resumed_final[name]), name
        assert (prov.sha256_state_dict(reference_final)
                == prov.sha256_state_dict(resumed_final))


def test_stop_after_is_a_resumable_pilot_on_the_full_schedule():
    """Pause at the gate without training a different short-horizon experiment."""
    with tempfile.TemporaryDirectory(prefix="e6a_stop_after_",
                                     ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        reference = _run(tmp / "reference", max_steps=20, resume="none")
        reference_final = _weights(reference.run_dir, 20)

        paused = _run(tmp / "paused", max_steps=20, resume="none", stop_after=10)
        assert (paused.run_dir / "checkpoints" / "step_10").exists()
        run_config = json.loads(
            (paused.run_dir / "run_config.json").read_text(encoding="utf-8"))
        assert run_config["max_steps"] == 20

        resumed = _run(tmp / "paused", max_steps=20, resume="auto")
        resumed_final = _weights(resumed.run_dir, 20)
        for name in reference_final:
            assert torch.equal(reference_final[name], resumed_final[name]), name


def test_the_schedule_horizon_is_part_of_the_run():
    """Guards the assumption the test above rests on.

    Two runs that differ only in ``max_steps`` must NOT agree at a shared step — the cosine
    schedule depends on the horizon. If they ever did agree, the resume test's simulated
    interruption would be testing nothing.
    """
    with tempfile.TemporaryDirectory(prefix="e6a_horizon_",
                                     ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        long_run = _run(tmp / "long", max_steps=20, resume="none")
        short_run = _run(tmp / "short", max_steps=10, resume="none")
        long_10 = _weights(long_run.run_dir, 10)
        short_10 = _weights(short_run.run_dir, 10)
        assert any(not torch.equal(long_10[k], short_10[k]) for k in long_10)


def test_resuming_under_a_different_horizon_is_refused():
    """The pilot -> Phase 2 workflow would otherwise splice two cosines together.

    ``LambdaLR.state_dict()`` excludes the lambda by design, so ``load_checkpoint``
    restores ``last_epoch`` but not the schedule it came from. The lambda closes over
    ``max_steps``, so resuming a 2,000-step pilot into the 10,000-step production run — the
    documented sequence, same config, same run directory — would follow neither schedule:
    the LR jumps at the resume point. ``test_the_schedule_horizon_is_part_of_the_run``
    above proves the horizon really does change the trajectory, which is what makes this a
    correctness issue rather than a cosmetic one.
    """
    import pytest

    with tempfile.TemporaryDirectory(prefix="e6a_horizon_guard_",
                                     ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "run"
        _run(root, max_steps=10, resume="none")

        with pytest.raises(SystemExit) as excinfo:
            _run(root, max_steps=20, resume="auto")
        message = str(excinfo.value)
        assert "max_steps=10" in message and "max_steps=20" in message
        assert "LambdaLR" in message, "the message must say WHY, not just refuse"
        assert "--resume none" in message, "and it must say how to proceed"


def test_resuming_at_the_recorded_horizon_still_works():
    """The guard must not break the ordinary resume it sits in front of."""
    with tempfile.TemporaryDirectory(prefix="e6a_horizon_ok_",
                                     ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "run"
        first = _run(root, max_steps=20, resume="none")
        shutil.rmtree(first.run_dir / "checkpoints" / "step_20")
        second = _run(root, max_steps=20, resume="auto")
        assert (second.run_dir / "checkpoints" / "step_20").exists()


def test_resume_none_bypasses_the_horizon_guard():
    """`--resume none` is the documented escape hatch and must actually work."""
    with tempfile.TemporaryDirectory(prefix="e6a_horizon_none_",
                                     ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "run"
        _run(root, max_steps=10, resume="none")
        setup = _run(root, max_steps=20, resume="none")     # must not raise
        assert int(json.loads((setup.run_dir / "run_config.json")
                              .read_text(encoding="utf-8"))["max_steps"]) == 20


def test_step_zero_is_saved_before_the_first_update():
    """``03`` §1.6: an explicit save before the loop, not a branch inside it.

    A step-0 branch inside the loop would, on resume, save *post*-update weights under the
    name ``step_0`` — and the E6B drift baseline is measured against exactly that file.
    """
    with tempfile.TemporaryDirectory(prefix="e6a_step0_",
                                     ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        setup = _run(tmp / "run", max_steps=10, resume="none")

        # Compared against a freshly constructed student rather than against
        # `initial_state_sha256`: `save_pretrained` drops tied weights (GPT-Neo ties
        # lm_head to wte), so the file legitimately has fewer keys than the in-memory
        # state dict whose sha256 the run records.
        step0 = _weights(setup.run_dir, 0)
        fresh = td.build_student(setup.student_config, 0).state_dict()
        assert set(step0) <= set(fresh)
        for name, tensor in step0.items():
            assert torch.equal(tensor, fresh[name]), name

        state = json.loads((setup.run_dir / "checkpoints" / "step_0" /
                            "trainer_state.json").read_text(encoding="utf-8"))
        assert state["pre_first_update"] is True
        assert state["step"] == 0

        # And training actually moved the weights, so the comparison above has content.
        step10 = _weights(setup.run_dir, 10)
        assert any(not torch.equal(step0[k], step10[k]) for k in step0)


def test_resume_restores_optimiser_scheduler_and_rng_state():
    with tempfile.TemporaryDirectory(prefix="e6a_state_",
                                     ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        setup = _run(tmp / "run", max_steps=10, resume="none")
        checkpoint = setup.run_dir / "checkpoints" / "step_10"
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pt",
                     "checkpoint_sha256.txt", "model.safetensors"):
            assert (checkpoint / name).exists(), name

        digest = (checkpoint / "checkpoint_sha256.txt").read_text(encoding="utf-8").strip()
        assert digest == prov.sha256_tree(checkpoint,
                                          patterns=("*.safetensors", "*.json"))

        assert td.latest_checkpoint(setup.run_dir) == checkpoint


def test_resume_from_an_empty_directory_starts_at_zero():
    with tempfile.TemporaryDirectory(prefix="e6a_fresh_",
                                     ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        assert td.latest_checkpoint(tmp) is None
        setup = _run(tmp / "run", max_steps=10, resume="auto")
        assert (setup.run_dir / "checkpoints" / "step_0").exists()

        # Removing the checkpoints and resuming again re-runs from scratch, reproducibly.
        first = _weights(setup.run_dir, 10)
        shutil.rmtree(setup.run_dir / "checkpoints")
        again = _run(tmp / "run", max_steps=10, resume="auto")
        second = _weights(again.run_dir, 10)
        for name in first:
            assert torch.equal(first[name], second[name]), name
