"""Run ONE explicit E1/E2/E3 state and panel; never auto-launch another state."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from sinklab.mechanistic_admission import admit_job, load_model, runtime_identity
from sinklab.mechanistic_run import VERSION, read_json, run_state, verify_bundle, write_json, sha256_file
from sinklab.mechanism_trace import MechanisticGPT2Adapter
from sinklab.mechanism_injection import evaluation_mode
from sinklab.provenance import payload_digest


def engineering_fixture(phase, device):
    """Small but real 128-token GPT-2; synthetic doses/tolerances are labeled."""
    from transformers import GPT2Config, GPT2LMHeadModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(921)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=31, n_embd=24, n_head=3, n_layer=2,
            n_positions=128, n_ctx=128, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
            _attn_implementation="eager"))
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_attn.bias.normal_(0,.13)
    model = model.float().to(device).eval()
    items = [{"id": f"synthetic-{i}", "input_ids": [(k+i)%30+1 for k in range(128)],
              "attention_mask": [True]*length+[False]*(128-length)} for i,length in enumerate((128,117))]
    settings = {"atol": 2e-6, "rtol": 2e-5, "token_chunk": 16, "denominator_floor": 1e-8}
    if phase == "E1":
        settings.update(alphas=[0.,.25,.5,.75,1.],control_seed=17,scopes=[{"name":"native","layers":[0,1]}])
    else:
        settings.update(layers=[0,1])
    if phase == "E3":
        settings.update(etas=[0.,.01,.05], norm_floor=1e-10, control_seed=313,
            reference="clean_residual_input_before_ln_1", query_min=2, nonsink_keys=[1,2], orders=[[0,1],[1,0]])
    return model, items, settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--output",type=Path,required=True)
    prepare = commands.add_parser("prepare-panel")
    prepare.add_argument("--corpus",type=Path,required=True)
    prepare.add_argument("--registered-panels",type=Path,required=True)
    prepare.add_argument("--confirmation-ids",type=Path,required=True)
    prepare.add_argument("--output",type=Path,required=True)
    for name in ("engineering", "scientific"):
        p = commands.add_parser(name)
        p.add_argument("--phase",choices=("E1","E2","E3"),required=True)
        p.add_argument("--seed",type=int,required=True)
        p.add_argument("--device",required=True)
        p.add_argument("--output",type=Path,required=True)
        if name == "scientific":
            p.add_argument("--lock",type=Path,required=True)
            p.add_argument("--approved-sha256",required=True)
            p.add_argument("--state",required=True)
            p.add_argument("--panel",choices=("discovery","confirmation"),required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_bundle(args.output)
        print(f"VERIFIED {result['identity']['phase']} {len(result['items'])} items {len(result['operations'])} operations")
        return 0
    if args.command == "prepare-panel":
        from sinklab.mechanistic_panel import prepare_frozen_panel
        repo = Path(__file__).resolve().parents[1]
        resolved=args.output.resolve()
        if resolved.is_relative_to(repo) or any(p.lower()=="upstream" for p in resolved.parts):
            parser.error("prepared panel must be external to the repository/reference-only trees")
        panel=prepare_frozen_panel(artifact_document=read_json(repo/"protocols/artifact.lock.json"),
            corpus_reference={"path":str(args.corpus.resolve()),"sha256":sha256_file(args.corpus)},
            panels_reference={"path":str(args.registered_panels.resolve()),"sha256":sha256_file(args.registered_panels)},
            confirmation_ids=read_json(args.confirmation_ids))
        resolved.parent.mkdir(parents=True,exist_ok=True)
        write_json(resolved,panel)
        print(f"PREPARED document-disjoint panels: {resolved}; scientific approval still required")
        return 0
    if args.seed != 0:
        parser.error("D24: explicit seed0 required; no implicit seed campaign")
    repo = Path(__file__).resolve().parents[1]
    # Avoid output writes inside the checked repository or protected source trees.
    resolved = args.output.resolve()
    if resolved.is_relative_to(repo) or any(p.lower() == "upstream" for p in resolved.parts):
        parser.error("result output must be external to the repository and reference-only trees")
    if resolved.exists():
        parser.error("output already exists; use a fresh attempt directory")
    runtime = runtime_identity(repo,args.device)
    if args.command == "engineering":
        model, items, settings = engineering_fixture(args.phase,args.device)
        identity = {"schema_version":VERSION,"phase":args.phase,"study":f"mechanistic_followup/{args.phase}",
            "seed":args.seed,"state":"synthetic_random_gpt2","panel":"synthetic","context_length":128,
            "run_id":"engineering-"+resolved.name,"engineering_only":True,"runtime":runtime,
            "panel_sha256":payload_digest({"items":items}),"protocol_sha256":None,"source":None}
        teacher_provider = None
    else:
        identity, settings, items, source, teacher_source = admit_job(read_json(args.lock),
            approved_sha256=args.approved_sha256,phase=args.phase,state=args.state,panel_name=args.panel,
            seed=args.seed,repo=repo,device=args.device)
        # No load occurs until ALL admission gates have passed.
        model = load_model(source,args.device)
        teacher = model if source["role"] == "teacher" else load_model(teacher_source,args.device)
        def teacher_provider(item):
            ids = torch.tensor([item["input_ids"]],dtype=torch.long,device=args.device)
            mask = torch.tensor([item["attention_mask"]],dtype=torch.bool,device=args.device)
            with evaluation_mode(teacher):
                return teacher(input_ids=ids,attention_mask=mask,use_cache=False).logits
    print(f"{args.phase} seed{args.seed} {identity['state']} {identity['run_id']} device={args.device} engineering_only={identity['engineering_only']}",flush=True)
    summary = run_state(MechanisticGPT2Adapter(model),items,phase=args.phase,settings=settings,
                        identity=identity,output=resolved,teacher_provider=teacher_provider)
    print(f"COMPLETE {len(summary['items'])} items {len(summary['operations'])} operations: {resolved}",flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
