"""E0 only: verify all recorded S4 items and write a separate descriptive bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sinklab.mechanistic_e0 import (CHANNELS, CONDITIONS, PROBES, STEPS, VERSION, TRAJECTORY_FIELDS, CONTEXT_FIELDS,
    analyze, audit_source, csv_bytes, digest_file, require, trajectory_context, validate_bundle, verify_bundle, write_new)
from sinklab.provenance import canonical_json_bytes, seal_payload, _no_duplicate_keys


def plot_data(rows: list[dict]) -> dict:
    return {"steps": list(STEPS), "x_axis": "optimizer updates (symlog, linthresh=100)",
        "metrics": ["delta_sink", "delta_ce_nats", "self_kl_nats"], "rows": [
            {k: row[k] for k in ("model_role", "condition", "step", "probe_id", "delta_sink", "delta_ce_nats", "self_kl_nats")}
            for row in rows], "step0": "random student initialization", "teacher": "fixed native36 reference"}


def plot_routes(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(PROBES), 3, figsize=(17, 30), constrained_layout=True)
    metrics = ("delta_sink", "delta_ce_nats", "self_kl_nats")
    for index, probe in enumerate(PROBES):
        reference = next(r for r in rows if r["model_role"] == "teacher" and r["probe_id"] == probe)
        for column, metric in enumerate(metrics):
            ax = axes[index, column]
            for condition in CONDITIONS:
                values = sorted((r for r in rows if r["condition"] == condition and r["probe_id"] == probe), key=lambda r: r["step"])
                ax.plot(STEPS, [r[metric] for r in values], marker="o", markersize=3, label=condition)
            ax.axhline(reference[metric], color="black", linestyle="--", label="teacher native36")
            ax.set_xscale("symlog", linthresh=100)
            ax.set_xticks(STEPS, [str(s) for s in STEPS])
            ax.set_title(probe, fontsize=9)
            ax.set_ylabel(f"{metric}\n{CHANNELS[metric]}", fontsize=8)
            ax.set_xlabel("optimizer updates", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("E0: recorded route responses; step0 is random initialization\n"
                 "Student native24 / teacher native36; scope and dose differences remain", fontsize=12)
    try:
        fig.savefig(output / "E0_ROUTE_TRAJECTORIES.svg", metadata={"Date": None})
        fig.savefig(output / "E0_ROUTE_TRAJECTORIES.png", dpi=110)
    finally:
        plt.close(fig)


def run(*, root: Path, independent_audit: Path, expected_audit_sha256: str,
        output: Path, panel_manifest: Path | None = None, recipe: dict | None = None, engineering_fixture: bool = False,
        s5_bundle: Path | None = None, expected_s5_audit_sha256: str | None = None,
        progress=None) -> dict:
    root, independent_audit = root.resolve(), independent_audit.resolve()
    output = output.resolve()
    require(not (output == REPO or REPO in output.parents), "output must be outside repository")
    # Walk ancestors to catch a different Git checkout too.
    require(not any((p / ".git").exists() for p in (output, *output.parents)), "output must be outside every Git tree")
    for source in (root, independent_audit.parent, *((panel_manifest.resolve().parent,) if panel_manifest else ()),
                   *((s5_bundle.resolve(),) if s5_bundle else ())):
        require(not (output == source or source in output.parents or output in source.parents), "output must be separate from source")
    require(not output.exists(), "output must be a new directory; completed/failing outputs are preserved")
    output.mkdir(parents=True)
    try:
        rows, provenance = audit_source(root, independent_audit, expected_audit_sha256,
            panel_manifest=panel_manifest, engineering_fixture=engineering_fixture, progress=progress)
        require(not output.is_relative_to(Path(provenance["frozen_panel_path"]).parent), "output overlaps frozen panel inputs")
        analysis = analyze(rows, recipe)
        context, context_provenance = trajectory_context(rows, provenance, s5_bundle, expected_s5_audit_sha256)
        provenance["S5_context"] = context_provenance
        code = {str(path.relative_to(REPO)): digest_file(path) for path in (
            REPO / "scripts/report_mechanistic_e0.py", REPO / "src/sinklab/mechanistic_e0.py",
            REPO / "src/sinklab/provenance.py", REPO / "src/sinklab/followup_policy.py")}
        source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            capture_output=True, text=True).stdout.strip()
        provenance.update(analysis_source_commit=source_commit, execution_source_file_sha256=code,
            created_utc=datetime.now(timezone.utc).isoformat(), fingerprint_recipe=recipe,
            implementation_namespace="mechanistic_followup/E0")
        source_before = {name: digest_file(root / name) for name in ("SHA256SUMS.txt", "S4_FINAL_AUDIT.json", "S4_RUN_MANIFEST.json")}
        write_new(output / "E0_PROVENANCE.json", canonical_json_bytes(seal_payload(provenance)) + b"\n")
        write_new(output / "E0_ANALYSIS.json", canonical_json_bytes(seal_payload({"rows": rows, "analysis": analysis,
            "recipe": recipe, "trajectory_context": context})) + b"\n")
        write_new(output / "E0_S4_ROUTE_TRAJECTORIES.csv", csv_bytes(rows, TRAJECTORY_FIELDS))
        comparison_fields = list(analysis["comparisons"][0])
        write_new(output / "E0_TEACHER_STUDENT_FINGERPRINTS.csv", csv_bytes(analysis["comparisons"], comparison_fields))
        write_new(output / "E0_TRAJECTORY_CONTEXT.csv", csv_bytes(context, CONTEXT_FIELDS))
        write_new(output / "E0_PLOT_DATA.json", canonical_json_bytes(plot_data(rows)) + b"\n")
        plot_routes(rows, output)
        require(all(digest_file(root / name) == digest for name, digest in source_before.items()), "source receipts changed during outputs")
        artifact_hashes = {p.name: digest_file(p) for p in sorted(output.iterdir())}
        write_new(output / "SHA256SUMS.txt", "".join(f"{digest}  {name}\n" for name, digest in artifact_hashes.items()).encode())
        artifact_hashes["SHA256SUMS.txt"] = digest_file(output / "SHA256SUMS.txt")
        receipt = {"version": VERSION, "status": "COMPLETE", "files": artifact_hashes,
            "engineering_only": engineering_fixture, "scientific_record_reanalysis": not engineering_fixture,
            "training": False, "new_inference": False, "source_files_unchanged": True,
            "fresh_records_verified": provenance["fresh_records_verified"], "summary_count": 36,
            "trajectory_rows": len(rows), "component_comparison_rows": len(analysis["comparisons"]),
            "composite_status": analysis["composite_status"], "analysis_source_commit": source_commit,
            "S5_context_status": context_provenance["status"],
            "execution_source_file_sha256": code, "independent_audit_file_sha256": expected_audit_sha256}
        validate_bundle(output, receipt, completion_present=False)
        temporary = output / "COMPLETE.tmp"
        write_new(temporary, canonical_json_bytes(seal_payload(receipt)) + b"\n")
        temporary.rename(output / "COMPLETE.json")
        return receipt
    except Exception as exc:
        status = "BLOCKED" if isinstance(exc, (FileNotFoundError, PermissionError)) else "FAILED"
        if not (output / "COMPLETE.json").exists():
            write_new(output / f"{status}.json", canonical_json_bytes(seal_payload({"version": VERSION,
                "status": status, "error_type": type(exc).__name__, "error": str(exc),
                "engineering_only": engineering_fixture, "training": False, "new_inference": False})) + b"\n")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    check = modes.add_parser("verify", help="verify a completed E0 bundle without scientific sources")
    check.add_argument("--output", type=Path, required=True)
    audit = modes.add_parser("audit", help="read and verify existing S4 records; no inference")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--independent-audit", type=Path, required=True)
    audit.add_argument("--expected-audit-sha256", required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--fingerprint-recipe", type=Path)
    audit.add_argument("--panel-manifest", type=Path, help="relocated original frozen panel, with exact original hashes")
    audit.add_argument("--s5-bundle", type=Path, help="optional already-recorded attention-deletion trajectory context")
    audit.add_argument("--expected-s5-audit-sha256", help="required pin when --s5-bundle is supplied")
    audit.add_argument("--engineering-fixture", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_bundle(args.output)
        else:
            recipe = json.loads(args.fingerprint_recipe.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys) if args.fingerprint_recipe else None
            result = run(root=args.root, independent_audit=args.independent_audit,
                expected_audit_sha256=args.expected_audit_sha256, output=args.output,
                panel_manifest=args.panel_manifest, recipe=recipe, engineering_fixture=args.engineering_fixture,
                s5_bundle=args.s5_bundle, expected_s5_audit_sha256=args.expected_s5_audit_sha256,
                progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED" if isinstance(exc, (FileNotFoundError, PermissionError)) else "FAILED",
            "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
