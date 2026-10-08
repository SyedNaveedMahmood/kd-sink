"""Real CLI/plot roundtrip on engineering fixtures, with network/model access absent."""

import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_mechanistic_e0 import source_fixture

ROOT = Path(__file__).resolve().parents[2]


def command(arguments, *, env=None):
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", UV_OFFLINE="1")
    environment.update(env or {})
    return subprocess.run([sys.executable, str(ROOT / "scripts/report_mechanistic_e0.py"), *arguments],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=120)


def test_cli_generates_real_figures_and_independently_verifies(tmp_path):
    root, audit, pin = source_fixture(tmp_path, items=1)
    output = tmp_path / "outputs" / "e0"
    result = command(["audit", "--root", str(root), "--independent-audit", str(audit),
        "--expected-audit-sha256", pin, "--output", str(output), "--engineering-fixture"])
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert lines[0]["event"] == "e0_source_audit"
    assert lines[-1]["status"] == "COMPLETE" and lines[-1]["scientific_record_reanalysis"] is False
    assert lines[-1]["fresh_records_verified"] == 360
    assert ET.parse(output / "E0_ROUTE_TRAJECTORIES.svg").getroot().tag.endswith("svg")
    assert (output / "E0_ROUTE_TRAJECTORIES.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    plot = json.loads((output / "E0_PLOT_DATA.json").read_text())
    assert len(plot["rows"]) == 360 and plot["steps"] == [0, 100, 500, 2000, 10000]
    # Source is no longer needed to verify the derived bundle.
    root.rename(tmp_path / "archived_source")
    verified = command(["verify", "--output", str(output)])
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["status"] == "COMPLETE"


def test_cli_requires_pinned_audit_and_has_no_default_launch(tmp_path):
    assert command([]).returncode == 2
    assert command(["audit", "--root", str(tmp_path)]).returncode == 2
    help_result = command(["--help"])
    assert help_result.returncode == 0 and "no inference" in help_result.stdout


def test_audit_import_and_execution_do_not_load_model_stack(tmp_path):
    root, audit, pin = source_fixture(tmp_path, items=1)
    code = "from sinklab.mechanistic_e0 import audit_source; from pathlib import Path; import sys; "
    code += f"audit_source(Path({str(root)!r}),Path({str(audit)!r}),{pin!r},engineering_fixture=True); "
    code += "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; print('model_stack_absent')"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1"}
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment,
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0 and result.stdout.strip() == "model_stack_absent", result.stderr
