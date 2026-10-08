"""Produce approval-ready drafts; this command cannot approve a protocol."""
import argparse
import subprocess
from pathlib import Path

import torch

from sinklab.calibrated_probes import require, validate_panel_split
from sinklab.followup_policy import D24_SHA256
from sinklab.mechanistic_admission import E1_GRID, E2_GRID, runtime_identity, admit_job
from sinklab.mechanistic_run import VERSION, read_json, write_json, sha256_file, validate_settings
from sinklab.provenance import seal_payload, verify_envelope, payload_digest
from sinklab.training_entry import _teacher_map


def phase_settings(phase, state):
    layers = 36 if state == "teacher" else 24
    settings = {"atol": 1e-3, "rtol": 1e-4, "token_chunk": 16, "denominator_floor": 1e-8,
        "geometry_atol": 1e-10, "geometry_rtol": 0.}
    if phase == "E1":
        scopes = [{"name": "native", "layers": list(range(layers))}]
        if state == "teacher": scopes.append({"name": "mapped_teacher", "layers": list(_teacher_map("S1"))})
        settings.update(alphas=[0., .25, .5, .75, 1.], control_seed=20260927, scopes=scopes)
    else: settings["layers"] = list(range(layers))
    if phase == "E3":
        settings.update(layers=[5, 17, 29] if state == "teacher" else [3, 11, 19],
            etas=[0., .01, .03, .10], norm_floor=1e-8, control_seed=20260927,
            norm_atol=1e-6, norm_rtol=1e-6,
            reference="clean_residual_input_before_ln_1", query_min=2, nonsink_keys=[1, 2],
            orders=[list(range(layers)), list(reversed(range(layers)))])
    validate_settings(phase, settings, layers)
    return settings


def build(*, repo, artifact_root, panel_dir, output, device, edition, qualification_root=None):
    require(not output.exists() and not output.resolve().is_relative_to(repo), "fresh external candidate directory required")
    artifact, artifact_sha = verify_envelope(read_json(repo/"protocols/artifact.lock.json"))
    inventory_path = Path("D:/KD-SINK-central/analysis/stage08_scientific_20261005/S4_attempt02/S4_RUN_MANIFEST.json")
    require(sha256_file(inventory_path) == "794bf1673c46c00a190e74bea2f2ec1e1468bf9b5d9420090244d87de225da47", "original source inventory pin")
    inventory, _ = verify_envelope(read_json(inventory_path))
    panel = read_json(panel_dir/"panel.json")
    split = validate_panel_split(panel["discovery"], panel["confirmation"])
    require(split["discovery_items"] == split["confirmation_items"] == 300, "candidate panel count differs")
    panel_receipt, _ = verify_envelope(read_json(panel_dir/"PANEL_PREPARATION.json"))
    require(panel_receipt["files"]["panel.json"] == sha256_file(panel_dir/"panel.json") and not panel_receipt["researcher_approval"], "panel candidate receipt mismatch")
    sources = {"teacher": {"model_id": artifact["teacher"]["id"], "revision": artifact["teacher"]["revision"],
        "config": {"path": str(artifact_root/"teacher/config.json"), "sha256": artifact["teacher"]["config_sha256"]},
        "weights": {"path": str(artifact_root/"teacher/model.safetensors"), "sha256": artifact["teacher"]["weights_sha256"]}}}
    for state in E2_GRID:
        condition, suffix = state.split("/"); step = int(suffix.removeprefix("step"))
        original = inventory["source_inventory"]["sources"][condition]
        checkpoint = next(c for c in original["checkpoints"] if c["step"] == step)
        sources[state] = {"initialization": "configuration_random", "identity": original["identity"],
            "config": {"path": str(artifact_root/"student-config/config.json"), "sha256": artifact["student_config"]["sha256"]},
            "weights": {"path": checkpoint["model_path"], "sha256": checkpoint["model_file_sha256"]},
            "manifest": {"path": checkpoint["manifest_path"], "sha256": checkpoint["manifest_file_sha256"]}}
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    runtime = runtime_identity(repo, device)
    specifications = {str(p.relative_to(repo)): sha256_file(p) for p in sorted((repo/"design/e6a_v2/studies").glob("MECHANISTIC_*_v1.md"))}
    require(len(specifications) == 4, "exact phase/common specification set required")
    output.mkdir(parents=True)
    for phase in ("E1", "E2", "E3"):
        grid = ["teacher", *(E1_GRID if phase == "E1" else E2_GRID)]
        settings = {state: phase_settings(phase, state) for state in grid}
        qualification = {"path": None, "sha256": None}
        if qualification_root:
            path = qualification_root/f"{phase}_QUALIFICATION.json"
            measured = read_json(path)
            require(measured["runtime"] == runtime and measured["settings_sha256"] == payload_digest(settings), "qualification/runtime/settings mismatch")
            qualification = {"path": str(path), "sha256": sha256_file(path)}
        payload = {"schema_version": VERSION, "candidate_version": edition, "status": "draft", "production_ready": False,
            "approval": {"authority": None, "record": None}, "phase": phase, "seed": 0, "d24_sha256": D24_SHA256,
            "grid": grid, "sources": {state: sources[state] for state in grid}, "teacher_layer_map": list(_teacher_map("S1")),
            "panel": {"path": str(panel_dir/"panel.json"), "sha256": sha256_file(panel_dir/"panel.json")},
            "panel_counts": {"discovery": 300, "confirmation": 300}, "runtime": runtime, "settings": settings,
            "qualification": qualification, "run_id": f"mechanistic-{phase.lower()}-seed0-v1",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "specification_files_sha256": specifications, "artifact_lock_sha256": artifact_sha,
            "selection_receipt": {"path": str(panel_dir/"selection.json"), "sha256": sha256_file(panel_dir/"selection.json")},
            "analysis_rules": {"primary": ["delta_ce_nats", "self_kl_nats"], "reporting": "descriptive paired records, no CI/p-values/equivalence",
                "mixed_unit_composite": "disabled", "matched_dose_metric": "sink_removed_fraction", "matched_dose_targets": [.10, .25, .50],
                "geometry_absolute_tolerance_proposal": 1e-10, "norm_absolute_tolerance_proposal": 1e-6, "norm_relative_tolerance_proposal": 1e-6},
            "pending_researcher_decisions": ["confirmation count300/source-only LM2000 rule", "descriptive statistics/disabled mixed-unit composite",
                "control seed20260927 and floors1e-8", "E1 matched-dose targets.10/.25/.50", "E3 15-state grid, eta0/.01/.03/.10, coarse representatives/fine-extension rule",
                "fixed batch1/chunk16/TF32-disabled Adrita runtime and10-percent VRAM headroom", "production numerical thresholds justified by engineering evidence"]}
        if phase == "E3": payload["selection"] = {"method": "prespecified_coarse", "selected_states": grid,
            "selected_layers": {state: settings[state]["layers"] for state in grid}, "discovery_evidence": None,
            "fine_extension": "disabled; exact prospective discovery-only rule in hashed E3 specification; requires new derived lock"}
        document = seal_payload(payload)
        try:
            admit_job(document, approved_sha256=document["sha256"], phase=phase, state="teacher", panel_name="discovery", seed=0, repo=repo, device=device)
        except ValueError as error:
            require("draft or unapproved" in str(error), "unexpected candidate admission failure")
        else: raise ValueError("candidate unexpectedly admitted scientific inference")
        write_json(output/f"{phase}_{edition}.candidate.json", document)
        print(f"{phase} candidate envelope {document['sha256']} (draft; scientific admission rejected)", flush=True)
    write_json(output/"CANDIDATE_INVENTORY.json", {"status": "PROTOCOL_FREEZE_PENDING_RESEARCHER_APPROVAL",
        "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}, "runtime_sha256": payload_digest(runtime),
        "builder_script_sha256": sha256_file(Path(__file__)), "new_scientific_inference": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", required=True)
    parser.add_argument("--edition", choices=("v1", "v2"), required=True); parser.add_argument("--qualification-root", type=Path)
    args = parser.parse_args()
    build(repo=Path(__file__).resolve().parents[1], artifact_root=args.artifact_root, panel_dir=args.panel_dir,
        output=args.output, device=args.device, edition=args.edition, qualification_root=args.qualification_root)
