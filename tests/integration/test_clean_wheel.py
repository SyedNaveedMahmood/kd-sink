"""T00: install the wheel in a clean directory with no reference tree."""

import os
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path


def _run(args, *, cwd, env):
    result = subprocess.run(args, cwd=cwd, env=env, text=True,
                            capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_offline_wheel_without_reference_tree(tmp_path):
    root = Path(__file__).resolve().parents[2]
    isolated = tmp_path / "isolated_source"
    isolated.mkdir()
    shutil.copy2(root / "pyproject.toml", isolated / "pyproject.toml")
    shutil.copy2(root / "README.md", isolated / "README.md")
    shutil.copytree(root / "src", isolated / "src")
    assert not (isolated / "Upstream").exists()
    assert not (isolated / "upstream").exists()
    assert not any(path.is_symlink() for path in isolated.rglob("*"))

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PIP_NO_INDEX="1", UV_OFFLINE="1", HF_HUB_OFFLINE="1")
    _run([sys.executable, "-m", "pip", "wheel", ".", "--no-build-isolation",
          "--no-deps", "--wheel-dir", str(tmp_path / "dist")],
         cwd=isolated, env=env)
    wheels = list((tmp_path / "dist").glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        assert not any("upstream" in name.lower() for name in archive.namelist())
        assert any(name == "sinklab/config.py" for name in archive.namelist())
        for module in ("calibrated_probes", "mechanism_trace", "mechanism_injection",
                       "mechanistic_run", "mechanistic_admission", "mechanistic_panel"):
            assert f"sinklab/{module}.py" in archive.namelist()

    clean_env = tmp_path / "clean_env"
    venv.create(clean_env, with_pip=True)
    python = clean_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps",
          str(wheels[0])], cwd=tmp_path, env=env)
    smoke = _run([str(python), "-c",
                  "import sinklab; from sinklab.config import RunSpec; "
                  "from sinklab.mechanistic_e0 import VERSION; "
                  "assert VERSION == 'mechanistic-e0-v1'; print(sinklab.__file__)"],
                 cwd=tmp_path, env=env)
    assert str(clean_env).lower() in smoke.stdout.lower()

    # Exercise the installed mechanistic APIs with shared pinned dependencies,
    # without processing the development environment's editable .pth files.
    # The package itself must resolve from the fresh wheel installation.
    import torch
    dependency_site = Path(torch.__file__).resolve().parents[1]
    program = r'''
import pathlib, socket, sys
import sinklab
installed = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(sinklab.__file__).resolve().is_relative_to(installed)
sys.path.append(sys.argv[2])
def blocked(*args, **kwargs):
    raise AssertionError("network access in installed-wheel smoke")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
import torch
torch.random.default_generator.manual_seed(923)
from transformers import GPT2Config, GPT2LMHeadModel
from sinklab.mechanism_trace import MechanisticGPT2Adapter
from sinklab.mechanistic_run import run_state, verify_bundle, VERSION
model = GPT2LMHeadModel(GPT2Config(vocab_size=19, n_embd=16, n_head=2,
    n_layer=2, n_positions=128, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
    _attn_implementation="eager")).float().eval()
items = [{"id":"installed-wheel", "input_ids":[i%18+1 for i in range(128)],
          "attention_mask":[True]*117+[False]*11}]
for phase in ("E1", "E2", "E3"):
    settings = {"atol":2e-6, "rtol":2e-5, "token_chunk":16, "denominator_floor":1e-8}
    if phase == "E1":
        settings.update(alphas=[0.,.25,.5,.75,1.], control_seed=17,
                        scopes=[{"name":"native", "layers":[0,1]}])
    else:
        settings.update(layers=[0,1])
    if phase == "E3":
        settings.update(etas=[0.,.01], norm_floor=1e-10, control_seed=313,
            reference="clean_residual_input_before_ln_1", query_min=2,
            nonsink_keys=[1,2], orders=[[0,1],[1,0]])
    identity = {"schema_version":VERSION, "phase":phase, "seed":0,
        "state":"synthetic-installed-wheel", "run_id":"wheel-smoke-"+phase,
        "engineering_only":True, "context_length":128}
    output = pathlib.Path(sys.argv[3])/phase
    result = run_state(MechanisticGPT2Adapter(model), items, phase=phase,
                       settings=settings, identity=identity, output=output)
    assert verify_bundle(output) == result
    assert all(row["behavior"]["valid_targets"] == 116
               for row in result["operations"].values())
print("INSTALLED_WHEEL E1 E2 E3 VERIFIED; network blocked")
'''
    actual = _run([str(python), "-c", program, str(clean_env),
                   str(dependency_site), str(tmp_path / "wheel_artifacts")],
                  cwd=tmp_path, env=env)
    assert "INSTALLED_WHEEL E1 E2 E3 VERIFIED; network blocked" in actual.stdout
