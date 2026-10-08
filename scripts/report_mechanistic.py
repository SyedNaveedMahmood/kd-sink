"""Read-only reaggregation and complete-grid joining for E1/E2/E3 bundles."""
from __future__ import annotations

import argparse
from pathlib import Path

from sinklab.calibrated_probes import observed_dose_match, require
from sinklab.mechanistic_admission import E1_GRID, E2_GRID
from sinklab.mechanistic_run import read_json, verify_bundle, write_json, sha256_file
from sinklab.provenance import verify_envelope, payload_digest


def complete_join(bundles, *, lock, approved_sha256, panel_name):
    payload,digest = verify_envelope(lock)
    require(digest == approved_sha256 and payload["status"] == "approved" and payload["production_ready"] is True,
            "explicit approved analysis protocol required")
    phase = payload["phase"]
    expected = set(payload["grid"])
    if phase in {"E1","E2"}:
        require(expected == {"teacher",*(E1_GRID if phase=="E1" else E2_GRID)}, "declared grid differs")
    else:
        require(phase == "E3" and "teacher" in expected and expected.issubset({"teacher",*E2_GRID}), "invalid E3 grid")
    results = {}
    for root in bundles:
        summary = verify_bundle(root); identity = summary["identity"]; state = identity["state"]
        require(state in expected and state not in results and identity["phase"] == phase and identity["seed"] == 0 and
                identity["panel"] == panel_name and identity["engineering_only"] is False and
                identity["protocol_sha256"] == digest and identity["panel_sha256"] == payload["panel"]["sha256"] and
                identity["runtime_sha256"] == payload_digest(payload["runtime"]) and
                len(summary["items"]) == payload["panel_counts"][panel_name] and
                summary["settings"] == payload["settings"][state], "join identity/duplicate/scope mismatch")
        source = payload["sources"][state]
        require(identity["source"]["weights_sha256"] == source["weights"]["sha256"] and
                identity["source"]["config_sha256"] == source["config"]["sha256"] and
                identity["source"]["source_identity"] == source.get("identity"), "join source binding mismatch")
        results[state] = {"bundle":str(Path(root).resolve()),"manifest_sha256":sha256_file(Path(root)/"manifest.json"),"summary":summary}
    require(set(results) == expected and len({tuple(r["summary"]["items"]) for r in results.values()}) == 1,
            "incomplete grid or mismatched item pairing")
    return {"phase":phase,"panel":panel_name,"protocol_sha256":digest,"states":results,
            "interpretation":"descriptive seed0, no across-seed replication or automatically classified route inheritance"}


def dose_comparisons(join, *, metric, targets):
    """Teacher mapped24 versus student native24; no matching on loss."""
    require(join["phase"] == "E1", "dose comparisons require E1")
    field = {"sink_removed_fraction":"sink_removed_fraction_item_mean",
             "attention_output_delta_rms":"scope_attention_output_delta_rms_item_mean"}[metric]
    teacher = join["states"]["teacher"]["summary"]
    result = []
    for state, source in join["states"].items():
        if state == "teacher": continue
        student = source["summary"]
        for route in ("q_bias", "k_top3", *(f"k_random{i}" for i in range(5))):
            curves=[]
            for summary, scope in ((teacher,"mapped_teacher"),(student,"native")):
                curve = [{"alpha":alpha,metric:summary["operations"][f"{route}/{scope}/{alpha:g}"][field]}
                         for alpha in summary["settings"]["alphas"]]
                curves.append(curve)
            for target in targets:
                match = {"status":"dose_unavailable","dose_metric":metric,"target":target}
                if all(all(r[metric] is not None for r in c) for c in curves):
                    match = observed_dose_match(*curves,dose_metric=metric,target=target)
                result.append({"state":state,"route":route,"teacher_scope":"mapped_teacher","student_scope":"native",**match})
    return result


def plot_bundle(summary, prefix):
    """Standard exportable plots; no outcome thresholds or scientific verdict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prefix = Path(prefix)
    paths = [prefix.with_suffix(".png"),prefix.with_suffix(".svg")]
    require(not any(p.exists() for p in paths), "figure output already exists")
    settings, phase = summary["settings"],summary["identity"]["phase"]
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout="constrained")
    if phase=="E1":
        for scope in settings["scopes"]:
            name=scope["name"]
            for route in ("q_bias","k_top3"):
                rows=[summary["operations"][f"{route}/{name}/{a:g}"] for a in settings["alphas"]]
                label=f"{route} / {name}"
                for axis,values in zip(axes,([r["sink_fingerprint"]["probed_sink"] for r in rows],
                    [r["behavior"]["delta_ce_nats"] for r in rows],
                    [r["scope_attention_output_delta_rms_item_mean"] for r in rows]),strict=True):
                    axis.plot(settings["alphas"],values,marker="o",label=label)
        for axis,title in zip(axes,("Sink mass (scope mean)","Delta CE (nats/target)","Output delta RMS (scope)"),strict=True):
            axis.set(xlabel="alpha",ylabel=title)
    elif phase=="E2":
        layers=settings["layers"]
        factors=summary["diagnostics"]["factor_summary"]
        axes[0].plot(layers,[factors[str(i)]["second_half_queries"]["head_sink_mass_mean"] for i in layers],marker="o")
        axes[1].plot(layers,[factors[str(i)]["all_q_ge1"]["projected_delta_norm_mean"] for i in layers],marker="o")
        axes[2].plot(layers,[summary["operations"][f"delete/layer{i}"]["behavior"]["delta_ce_nats"] for i in layers],marker="o")
        for axis,title in zip(axes,("Second-half sink mass","Local projected delta L2","Isolated delta CE (nats/target)"),strict=True):
            axis.set(xlabel="Native layer",ylabel=title)
    else:
        for layer in settings["layers"]:
            for direction in ("sink","random","orthogonal","non_sink"):
                rows=[summary["operations"][f"injection/{direction}/layer{layer}/eta{eta:g}"] for eta in settings["etas"]]
                for axis,field in zip(axes,("delta_ce_nats","self_kl_nats","prediction_flip_fraction"),strict=True):
                    axis.plot(settings["etas"],[r["behavior"][field] if r["behavior"] else float("nan") for r in rows],
                              marker="o",label=f"L{layer} {direction}")
        for axis,title in zip(axes,("Delta CE (nats/target)","Self KL (nats/target)","Prediction flip fraction"),strict=True):
            axis.set(xlabel="eta (entering-residual relative norm)",ylabel=title)
    for axis in axes:
        axis.grid(alpha=.2)
    if phase!="E2": axes[-1].legend(fontsize=7)
    identity=summary["identity"]
    fig.suptitle(f"{phase}: {identity['state']} ({'engineering only' if identity['engineering_only'] else identity['panel']})")
    prefix.parent.mkdir(parents=True,exist_ok=True)
    for path in paths:
        with path.open("xb") as stream:
            fig.savefig(stream,format=path.suffix[1:],dpi=180)
    plt.close(fig)
    return {str(p):sha256_file(p) for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--approved-sha256")
    parser.add_argument("--panel",choices=("discovery","confirmation"))
    parser.add_argument("--dose-metric",choices=("sink_removed_fraction","attention_output_delta_rms"))
    parser.add_argument("--dose-target",type=float,action="append")
    parser.add_argument("--figure-prefix",type=Path,help="single-bundle PNG/SVG exports")
    args = parser.parse_args(argv)
    if args.lock:
        if not args.approved_sha256 or not args.panel: parser.error("join requires approval digest and panel")
        result = complete_join(args.bundle,lock=read_json(args.lock),approved_sha256=args.approved_sha256,panel_name=args.panel)
    else:
        if len(args.bundle)!=1: parser.error("multi-state joins require an explicit approved lock")
        result = {"bundle":str(args.bundle[0]),"summary":verify_bundle(args.bundle[0])}
    if args.dose_metric:
        if not args.lock or not args.dose_target: parser.error("dose matching requires approved complete E1 join and explicit targets")
        result["dose_matches"] = dose_comparisons(result,metric=args.dose_metric,targets=args.dose_target)
    resolved = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if resolved.is_relative_to(repo) or any(p.lower()=="upstream" for p in resolved.parts):
        parser.error("report must be external to the repository/reference-only trees")
    resolved.parent.mkdir(parents=True,exist_ok=True)
    if args.figure_prefix:
        prefix=args.figure_prefix.resolve()
        if args.lock or prefix.is_relative_to(repo) or any(p.lower()=="upstream" for p in prefix.parts):
            parser.error("figures require a single bundle and an external output prefix")
        result["figures"] = plot_bundle(result["summary"],prefix)
    write_json(resolved,result)
    print(f"VERIFIED report: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
