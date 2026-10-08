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
