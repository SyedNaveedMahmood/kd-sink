"""Freeze a candidate holdout using the authorized source-only LM2000 rule."""
import argparse
import json
import subprocess
from pathlib import Path

from sinklab.mechanistic_admission import _pinned
from sinklab.mechanistic_panel import select_disjoint_confirmation, panel_from_validated_corpus
from sinklab.mechanistic_run import read_json, write_json, sha256_file
from sinklab.owt_compat import load_owt_corpus, panels_from_validated_corpus
from sinklab.provenance import verify_envelope, seal_payload
from sinklab.calibrated_probes import require, validate_panel_split


def write_ids(path, ids):
    """The existing prepare-panel CLI requires a JSON list, not a lock object."""
    with path.open("xb") as stream:
        stream.write(json.dumps(ids, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def prepare(root, output, count):
    repo = Path(__file__).resolve().parents[1]
    require(not output.exists(), "fresh external output required")
    require(not output.resolve().is_relative_to(repo), "panel artifacts must be external")
    output.mkdir(parents=True)
    try:
        artifact, artifact_sha = verify_envelope(read_json(repo / "protocols/artifact.lock.json"))
        corpus_ref = {"path": str(root / "corpus" / f"owt-corpus-{artifact['corpus']['payload_sha256']}.json"), "sha256": artifact["corpus"]["file_sha256"]}
        panels_ref = {"path": str(root / "panels" / f"owt-panels-{artifact['panels']['payload_sha256']}.json"), "sha256": artifact["panels"]["file_sha256"]}
        print("Validating complete original corpus, packing, tokenizer and document ownership", flush=True)
        corpus_path, panels_path = _pinned(corpus_ref), _pinned(panels_ref)
        corpus, corpus_sha = load_owt_corpus(corpus_path, tokenizer_sha256=artifact["tokenizer"]["files_sha256"])
        registered, registered_sha = verify_envelope(read_json(panels_path))
        require(corpus_sha == artifact["corpus"]["payload_sha256"] and registered_sha == artifact["panels"]["payload_sha256"] and
                panels_from_validated_corpus(corpus, corpus_sha)["payload"] == registered, "original panel reconstruction differs")
        ids, selection = select_disjoint_confirmation(corpus, registered, count=count)
        panel = panel_from_validated_corpus(corpus, registered, ids, tokenizer_sha256=artifact["tokenizer"]["files_sha256"])
        panel["preparation"] = {"artifact_lock_sha256": artifact_sha, "corpus": corpus_ref, "registered_panels": panels_ref,
            "selection": "explicit confirmation block IDs; authority in separate selection receipt",
            "source_validation": "complete original OWT packing/tokenizer/document ownership contract"}
        require(sha256_file(corpus_path) == corpus_ref["sha256"] and sha256_file(panels_path) == panels_ref["sha256"], "source changed during preparation")
        write_ids(output / "confirmation_ids.json", ids)
        write_json(output / "selection.json", selection)
        write_json(output / "panel.json", panel)
        receipt = {"status": "candidate_prepared", "production_ready": False, "researcher_approval": False,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "source_script_sha256": sha256_file(Path(__file__)), "artifact_lock_sha256": artifact_sha,
            "split": validate_panel_split(panel["discovery"], panel["confirmation"]), "new_outcomes_used": False,
            "historical_exposure": "LM2000 has original S1 clean endpoint outcomes; no new route/delete/injection outcomes examined for selection",
            "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}}
        write_json(output / "PANEL_PREPARATION.json", seal_payload(receipt))
        print(receipt, flush=True)
    except Exception as error:
        write_json(output / "FAILED.json", {"error": type(error).__name__, "detail": str(error), "production_ready": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, required=True)
    args = parser.parse_args()
    prepare(args.artifact_root, args.output, args.candidate_count)
