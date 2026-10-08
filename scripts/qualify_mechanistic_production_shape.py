"""Engineering-only full-size qualification; never accepts scientific panels."""
import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from sinklab.calibrated_probes import RouteProbe, apply_route, observed_forward, transport, require
from sinklab.mechanism_trace import MechanisticGPT2Adapter, ParityTolerance, OutputEdit, isolated_parity
from sinklab.mechanism_injection import evaluation_mode, equal_norm_injections, loss_geometry, rescue_for_layer
from sinklab.mechanistic_admission import _source, load_model, runtime_identity, E2_GRID
from sinklab.mechanistic_run import read_json, write_json, sha256_file, run_state, verify_bundle, VERSION
from sinklab.probes import probe_plan, apply_probe, epe_directions
from sinklab.provenance import payload_digest, verify_envelope
from sinklab.training_entry import _model_digest


def normalized_uuid(value):
    return value.removeprefix("GPU-").lower()


def gpu_sample(uuid):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits"], text=True)
    rows = [line.split(",") for line in output.strip().splitlines()]
    matches = [[v.strip() for v in row] for row in rows if normalized_uuid(row[0].strip()) == normalized_uuid(uuid)]
    require(len(matches) == 1, "exact target GPU must be present once")
    ident, name, total, used, temperature, driver = matches[0]
    return {"uuid": ident, "name": name, "total_bytes": int(total)*2**20, "used_bytes": int(used)*2**20,
            "temperature_c": int(temperature), "driver": driver}


class Resources:
    def __init__(self, output, uuid, device):
        self.output, self.uuid, self.device = output, uuid, device
        self.samples, self.errors, self.stopped = [], [], threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        ram = psutil.virtual_memory()
        row = {"elapsed_seconds": time.perf_counter()-self.started, "gpu": gpu_sample(self.uuid),
            "ram_total_bytes": ram.total, "ram_available_bytes": ram.available,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "allocated_bytes": torch.cuda.memory_allocated(self.device), "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "disk_free_bytes": {drive: shutil.disk_usage(drive).free for drive in ("C:/", "D:/", "E:/")}}
        self.samples.append(row)
        with (self.output/"resource_samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True)+"\n")

    def loop(self):
        while not self.stopped.wait(1.):
            try: self.sample()
            except Exception as error: self.errors.append({"error": type(error).__name__, "detail": str(error)})

    def start(self):
        self.started = time.perf_counter(); self.sample(); self.thread.start()

    def stop(self):
        self.stopped.set(); self.thread.join(); self.sample()


class DistributionTolerance(ParityTolerance):
    """Same componentwise gate, plus all-element means and fixed-stride quantiles."""
    def __init__(self, atol, rtol):
        super().__init__(atol, rtol)
        object.__setattr__(self, "observations", [])

    def check(self, actual, reference, label):
        result = super().check(actual, reference, label)
        error = (actual.double()-reference.double()).abs().flatten()
        stride = max(1, math.ceil(error.numel()/65536))
        sample = error[::stride].cpu()
        quantiles = torch.quantile(sample, torch.tensor([.50,.95,.99], dtype=torch.float64))
        result.update(element_count=error.numel(), mean_absolute_error=float(error.mean()),
            rms_error=float(error.square().mean().sqrt()), quantile_sample_count=sample.numel(), quantile_stride=stride,
            sampled_p50=float(quantiles[0]), sampled_p95=float(quantiles[1]), sampled_p99=float(quantiles[2]),
            max_gate_fraction=float((error/(self.atol+self.rtol*reference.double().abs().flatten()).clamp_min(1e-300)).max()))
        self.observations.append(result)
        return result


def synthetic_items(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [{"id": f"engineering-uniform-{i}", "input_ids": torch.randint(1,50257,(128,),generator=generator).tolist(),
             "attention_mask": [True]*length+[False]*(128-length)} for i,length in enumerate((128,117))]


def hooks(model):
    return {id(m): (dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in model.modules()}


def rng_state():
    return (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone(),
            [r.clone() for r in torch.cuda.get_rng_state_all()])


def assert_rng_equal(before):
    after=rng_state()
    require(before[0]==after[0] and before[1][0]==after[1][0] and np.array_equal(before[1][1],after[1][1]) and
        before[1][2:]==after[1][2:] and torch.equal(before[2],after[2]) and
        len(before[3])==len(after[3]) and all(torch.equal(a,b) for a,b in zip(before[3],after[3])), "global RNG restoration failed")


class EngineeringFault(RuntimeError):
    pass


def exception_restoration(adapter, item, device):
    model=adapter.model
    ids=torch.tensor([item["input_ids"]],device=device);mask=torch.tensor([item["attention_mask"]],device=device,dtype=torch.bool)
    # Warm native HF lazy hooks before comparing our transient hooks.
    with evaluation_mode(model): adapter.forward(input_ids=ids,attention_mask=mask,use_cache=False)
    saved_hooks=hooks(model);saved_forwards=[b.attn.__dict__.get("forward") for b in model.transformer.h]
    saved_bias=[b.attn.c_attn.bias.detach().clone() for b in model.transformer.h]
    model.train();model.transformer.h[0].eval()
    modes=[m.training for m in model.modules()];before=rng_state()
    raised=False
    try:
        with evaluation_mode(model):
            with apply_route(model,RouteProbe("q_bias",1.,tuple(range(adapter.layer_count))),mask):
                random.random();np.random.rand();torch.rand(3);torch.rand(3,device=device)
                def fault(*_): raise EngineeringFault("deliberate production-shape restoration test")
                handle=model.transformer.h[1].register_forward_pre_hook(fault)
                try:
                    adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,),delete_layers=(0,),
                        output_edits={0:OutputEdit("add",torch.zeros(1,128,model.config.n_embd,device=device))})
                finally:handle.remove()
    except EngineeringFault: raised=True
    require(raised and [m.training for m in model.modules()]==modes, "exception/mode restoration failed")
    require(hooks(model)==saved_hooks and [b.attn.__dict__.get("forward") for b in model.transformer.h]==saved_forwards,
            "exception hook/attention-method restoration failed")
    require(all(torch.equal(b.attn.c_attn.bias,saved) for b,saved in zip(model.transformer.h,saved_bias)), "exception parameter restoration failed")
    assert_rng_equal(before);model.eval()
    return {"deliberate_exception": "caught", "parameters": "bitwise_restored", "hooks": "restored", "attention_methods": "restored",
            "mixed_child_modes": "restored", "python_numpy_cpu_cuda_rng": "restored"}


def route_diagnostics(adapter, item, settings, tolerance, device):
    model=adapter.model;width=model.config.n_embd
    ids=torch.tensor([item["input_ids"]],device=device);mask=torch.tensor([item["attention_mask"]],device=device,dtype=torch.bool)
    plan=probe_plan(model,control_seed=settings["control_seed"])
    routes=[("q_bias_all",RouteProbe("q_bias",1.,tuple(range(adapter.layer_count)))),
            ("position0_to1",RouteProbe("position0_to1",1.,())),("epe_transport_layer0",RouteProbe("epe_transport",1.,(0,)))]
    routes += [(name,RouteProbe("k_input",1.,tuple(range(adapter.layer_count)),tuple(plan[name]["coordinates"])))
               for name in ("k_top3_all",*(f"k_random{i}_all" for i in range(5)))]
    rows=[]
    with evaluation_mode(model):
        clean,observed=observed_forward(adapter,ids,mask)
        for name,probe in routes:
            original=model.transformer.h[0].attn.c_attn.weight.detach().clone()
            biases=[b.attn.c_attn.bias.detach().clone() for b in model.transformer.h]
            with apply_route(model,probe,mask):
                if probe.route=="q_bias":
                    require(all(not b.attn.c_attn.bias[:width].any() and torch.equal(b.attn.c_attn.bias[width:],saved[width:])
                        for b,saved in zip(model.transformer.h,biases)), "Q-only selectivity")
                    require(torch.equal(model.transformer.h[0].attn.c_attn.weight,original), "Q edit changed weights")
                if probe.route=="k_input":
                    expected=original.clone();expected[list(probe.coordinates),width:2*width]=0
                    require(torch.equal(model.transformer.h[0].attn.c_attn.weight,expected), "K-only first-layer full-matrix selectivity")
                    require(all(not b.attn.c_attn.weight[list(probe.coordinates),width:2*width].any() for b in model.transformer.h), "K selected slices not edited")
                    require(all(torch.equal(b.attn.c_attn.bias,saved) for b,saved in zip(model.transformer.h,biases)), "K edit changed biases")
                edited,_=observed_forward(adapter,ids,mask)
            with apply_probe(model,name,plan=plan,mask=mask):legacy=adapter.forward(input_ids=ids,attention_mask=mask,use_cache=False).logits
            require(torch.equal(edited.logits,legacy), "alpha1 legacy comparator not bitwise identical")
            rows.append({"route":name,"legacy_logit_parity":"bitwise_equal","max_absolute_error":0.})
            require(torch.equal(model.transformer.h[0].attn.c_attn.weight,original) and
                all(torch.equal(b.attn.c_attn.bias,saved) for b,saved in zip(model.transformer.h,biases)), "route parameter restore")
        directions=epe_directions(model);u0,u1=directions["u"];out=observed[0]["epe_output"]
        for alpha in settings["alphas"]:
            changed=transport(out,u0,u1,mask,alpha)
            require(torch.equal(changed[:,2:],out[:,2:]), "EPE changed unrelated positions")
            tolerance.check(changed[:,0]+changed[:,1],out[:,0]+out[:,1],"epe_sum_conservation")
        zero=RouteProbe("q_bias",0.,tuple(range(adapter.layer_count)))
        with apply_route(model,zero,mask):noop=adapter.forward(input_ids=ids,attention_mask=mask,use_cache=False).logits
        require(torch.equal(noop,clean.logits), "zero-alpha parity must be exact")
    return rows


def local_diagnostics(adapter, item, settings, tolerance, device):
    ids=torch.tensor([item["input_ids"]],device=device);mask=torch.tensor([item["attention_mask"]],device=device,dtype=torch.bool)
    rows=[]
    with evaluation_mode(adapter.model):
        for layer in range(adapter.layer_count):
            start=len(tolerance.observations)
            clean,parity=isolated_parity(adapter,ids,mask,layer,tolerance)
            rows.append({"layer":layer,"checks":tolerance.observations[start:]})
        for layer in settings["layers"]:
            clean=adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,))
            for eta in settings["etas"]:
                edits,metadata=equal_norm_injections(clean.traces[layer],mask,eta=eta,norm_floor=settings["norm_floor"],
                    control_seed=settings["control_seed"],query_min=settings["query_min"],nonsink_keys=tuple(settings["nonsink_keys"]))
                require(metadata["common_available_positions"]>0,"full-size fixture must exercise nondegenerate directions")
                common=torch.tensor(metadata["common_support_mask"],device=device,dtype=torch.bool)
                expected=eta*clean.traces[layer].residual_input.double().norm(dim=-1)[common]
                for direction,delta in edits.items():
                    require(not delta[~common].any(),"control support leak")
                    norm=delta.double().norm(dim=-1)[common]
                    norm_tolerance=DistributionTolerance(settings["norm_atol"],settings["norm_rtol"])
                    record=norm_tolerance.check(norm,expected,"equal_relative_norm_injection")
                    rows.append({"layer":layer,"eta":eta,"direction":direction,"norm_check":record})
                    if eta==0:
                        zero=adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,),output_edits={layer:OutputEdit("add",delta)})
                        require(torch.equal(zero.outputs.logits,clean.outputs.logits),"eta0 logits must be bitwise equal")
            rescue=rescue_for_layer(adapter,ids,mask,layer,tolerance=tolerance)
            tolerance.check(rescue["single_rescued"].outputs.logits,rescue["clean"].outputs.logits,"full_size_clean_output_rescue")
            geometry=loss_geometry(rescue["clean"].outputs.logits,rescue["all_deleted"].outputs.logits,ids,mask,
                tolerance=ParityTolerance(settings["geometry_atol"],settings["geometry_rtol"]),token_chunk=settings["token_chunk"])
            rows.append({"layer":layer,"full_vocabulary_geometry_max_error":geometry["identity_max_error"],"valid_targets":geometry["valid_targets"]})
    return rows


def qualify(args):
    repo=Path(__file__).resolve().parents[1];output=args.output.resolve();device=torch.device(args.device)
    require(device.type=="cuda" and device.index is not None,"fixed CUDA index required")
    require(not output.exists() and not output.is_relative_to(repo),"fresh external output required")
    require(args.seed==0 and args.student_state in E2_GRID,"explicit seed0 retained student required")
    torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=runtime_identity(repo,device)
    require(normalized_uuid(runtime["hardware"]["uuid"])==normalized_uuid(args.gpu_uuid),"CUDA device does not match target GPU")
    require(0<args.headroom_fraction<1,"explicit headroom fraction required")
    documents={phase:read_json(args.candidates/f"{phase}_v1.candidate.json") for phase in ("E1","E2","E3")}
    protocols={phase:verify_envelope(doc)[0] for phase,doc in documents.items()}
    require(all(p["status"]=="draft" and p["runtime"]==runtime for p in protocols.values()),"engineering candidate/runtime mismatch")
    output.mkdir(parents=True);resources=Resources(output,args.gpu_uuid,device);resources.start();started=time.perf_counter()
    peaks=[];phases={};diagnostics={}
    try:
        # Availability/hash verification covers the entire required fifteen-state union.
        master=protocols["E2"];sources={}
        for state,source in master["sources"].items():
            sources[state]=_source(source,state)
            print(json.dumps({"event":"engineering_source_verified","state":state,"weights_sha256":sources[state]["weights_sha256"]}),flush=True)
        artifact,_=verify_envelope(read_json(repo/"protocols/artifact.lock.json"))
        tokenizer_root=Path(master["sources"]["teacher"]["config"]["path"]).parent.parent/"tokenizer"
        for name,digest in artifact["tokenizer"]["file_sha256"].items():require(sha256_file(tokenizer_root/name)==digest,"pinned tokenizer differs")
        write_json(output/"SOURCE_AVAILABILITY.json",{"status":"PASS","sources":sources,"tokenizer":artifact["tokenizer"],"original_S1_S4_inventory":"794bf1673c46c00a190e74bea2f2ec1e1468bf9b5d9420090244d87de225da47"})
        items=synthetic_items(args.input_seed);write_json(output/"synthetic_inputs.json",{"items":items,"input_seed":args.input_seed,"registered_panel":False})
        torch.cuda.reset_peak_memory_stats(device);load_started=time.perf_counter()
        teacher=load_model(sources["teacher"],device);student=load_model(sources[args.student_state],device)
        torch.cuda.synchronize(device)
        loading={"seconds":time.perf_counter()-load_started,"allocated_peak":torch.cuda.max_memory_allocated(device),"reserved_peak":torch.cuda.max_memory_reserved(device)}
        peaks.append(loading)
        require((teacher.config.n_layer,teacher.config.n_head,teacher.config.n_embd)==(36,20,1280) and
            (student.config.n_layer,student.config.n_head,student.config.n_embd)==(24,16,1024),"actual full architectures required")
        parameter_hashes={"teacher":_model_digest(teacher),args.student_state:_model_digest(student)}
        def teacher_provider(item):
            ids=torch.tensor([item["input_ids"]],device=device);mask=torch.tensor([item["attention_mask"]],device=device,dtype=torch.bool)
            with evaluation_mode(teacher):return teacher(input_ids=ids,attention_mask=mask,use_cache=False).logits
        for state,model in (("teacher",teacher),(args.student_state,student)):
            adapter=MechanisticGPT2Adapter(model)
            diagnostics[state]={"exception_restoration":exception_restoration(adapter,items[0],device)}
            for phase in ("E1","E2","E3"):
                settings=protocols[phase]["settings"][state]
                identity={"schema_version":VERSION,"phase":phase,"study":f"mechanistic_followup/{phase}","seed":0,"state":state,
                    "panel":"engineering_uniform_tokens","context_length":128,"run_id":f"full-shape-{phase}-{state.replace('/','-')}",
                    "engineering_only":True,"runtime":runtime,"panel_sha256":payload_digest({"items":items}),
                    "protocol_sha256":documents[phase]["sha256"],"source":sources[state],"teacher_source":sources["teacher"]}
                name=f"{phase}_{'teacher' if state=='teacher' else 'student'}";phase_start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
                summary=run_state(adapter,items,phase=phase,settings=settings,identity=identity,output=output/name,teacher_provider=teacher_provider)
                verified=verify_bundle(output/name);torch.cuda.synchronize(device)
                require(verified==summary,"bundle reaggregation mismatch")
                stats={"seconds":time.perf_counter()-phase_start,"allocated_peak":torch.cuda.max_memory_allocated(device),"reserved_peak":torch.cuda.max_memory_reserved(device),
                    "operation_count":len(summary["operations"]),"item_count":len(items),"bundle_complete_sha256":sha256_file(output/name/"COMPLETE")}
                phases[name]=stats;peaks.append(stats);print(json.dumps({"event":"full_shape_phase_complete","bundle":name,**stats}),flush=True)
            torch.cuda.reset_peak_memory_stats(device);diagnostic_start=time.perf_counter()
            tolerance=DistributionTolerance(protocols["E2"]["settings"][state]["atol"],protocols["E2"]["settings"][state]["rtol"])
            diagnostics[state]["route_comparators"]=route_diagnostics(adapter,items[0],protocols["E1"]["settings"][state],tolerance,device)
            diagnostics[state]["local_items"]=[local_diagnostics(adapter,item,protocols["E3"]["settings"][state],tolerance,device) for item in items]
            diagnostics[state]["error_distributions"]=tolerance.observations
            peaks.append({"allocated_peak":torch.cuda.max_memory_allocated(device),"reserved_peak":torch.cuda.max_memory_reserved(device),"seconds":time.perf_counter()-diagnostic_start})
            require(_model_digest(model)==parameter_hashes[state],"full parameter tensor-content hash changed")
            print(json.dumps({"event":"full_shape_diagnostics_complete","state":state}),flush=True)
        resources.stop()
        require(not resources.errors,"resource monitoring had failures")
        samples=resources.samples;total=runtime["hardware"]["total_memory_bytes"];headroom=int(total*args.headroom_fraction)
        peak_reserved=max(p["reserved_peak"] for p in peaks);peak_allocated=max(p["allocated_peak"] for p in peaks)
        estimated_total=samples[0]["gpu"]["used_bytes"]+peak_reserved
        min_free=min(r["gpu"]["total_bytes"]-r["gpu"]["used_bytes"] for r in samples)
        require(estimated_total<=total-headroom and min_free>=headroom,"real-shape VRAM headroom gate failed")
        require(min(r["ram_available_bytes"] for r in samples)>=args.ram_headroom_gib*2**30 and
            min(r["disk_free_bytes"]["D:/"] for r in samples)>=args.disk_headroom_gib*2**30,"host RAM/output disk headroom failed")
        resource_summary={"wall_seconds":time.perf_counter()-started,"peak_allocated_bytes":peak_allocated,"peak_reserved_bytes":peak_reserved,
            "baseline_nv_used_bytes":samples[0]["gpu"]["used_bytes"],"conservative_peak_total_use_bytes":estimated_total,
            "minimum_sampled_gpu_free_bytes":min_free,"required_gpu_headroom_bytes":headroom,"temperature_max_c":max(r["gpu"]["temperature_c"] for r in samples),
            "ram_min_available_bytes":min(r["ram_available_bytes"] for r in samples),"process_peak_sampled_rss_bytes":max(r["process_rss_bytes"] for r in samples),
            "disk_min_free_bytes":{drive:min(r["disk_free_bytes"][drive] for r in samples) for drive in ("C:/","D:/","E:/")},"samples":len(samples),"psutil_version":psutil.__version__}
        write_json(output/"NUMERICAL_DIAGNOSTICS.json",diagnostics)
        write_json(output/"RESOURCE_SUMMARY.json",resource_summary)
        common={"status":"qualified","engineering_only":True,"new_scientific_panel_inference":False,"scientific_production_shape":True,
            "architectures":["gpt2-large","gpt2-medium"],"runtime":runtime,"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
            "qualification_script_sha256":sha256_file(Path(__file__)),"parity_passed":True,"headroom_passed":True,"resources":resource_summary,
            "loading":loading,"phases":phases,"synthetic_input_sha256":sha256_file(output/"synthetic_inputs.json"),
            "source_availability_sha256":sha256_file(output/"SOURCE_AVAILABILITY.json"),"numerical_diagnostics_sha256":sha256_file(output/"NUMERICAL_DIAGNOSTICS.json"),
            "representative_student_state":args.student_state,"checkpoint_grid_hashes_verified":15,"input_seed":args.input_seed,
            "approval_status":"numerical/scientific protocol candidate remains unapproved","gpu_driver":samples[0]["gpu"]["driver"]}
        for phase in ("E1","E2","E3"):
            write_json(output/f"{phase}_QUALIFICATION.json",{**common,"phase":phase,"settings_sha256":payload_digest(protocols[phase]["settings"]),
                "candidate_envelope_sha256":documents[phase]["sha256"]})
        files={p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob("*") if p.is_file()}
        write_json(output/"QUALIFICATION_COMPLETE.json",{"status":"PASSED","engineering_only":True,"files":files,"resources":resource_summary,
            "source_commit":common["source_commit"],"script_sha256":common["qualification_script_sha256"],"scientific_execution_authorized":False})
        print(json.dumps({"status":"PASSED","resources":resource_summary}),flush=True)
    except Exception as error:
        if resources.thread.is_alive():resources.stop()
        write_json(output/"QUALIFICATION_FAILED.json",{"status":"FAILED","error":type(error).__name__,"detail":str(error),"engineering_only":True,
            "resource_monitor_errors":resources.errors,"completed_phases":phases,"script_sha256":sha256_file(Path(__file__))})
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--candidates",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",required=True);parser.add_argument("--gpu-uuid",required=True)
    parser.add_argument("--seed",type=int,required=True);parser.add_argument("--student-state",required=True);parser.add_argument("--input-seed",type=int,required=True)
    parser.add_argument("--headroom-fraction",type=float,required=True);parser.add_argument("--ram-headroom-gib",type=float,required=True);parser.add_argument("--disk-headroom-gib",type=float,required=True)
    qualify(parser.parse_args())
