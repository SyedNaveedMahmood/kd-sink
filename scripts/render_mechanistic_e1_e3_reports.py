"""Render scientific reports from the verified, unrounded CPU exports."""
import argparse
import csv
from pathlib import Path

from summarize_mechanistic_e1_e2 import CONDITIONS, read, require, sha


def load_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-directory', type=Path, required=True)
    args = parser.parse_args(); data = args.data_directory; repo = Path(__file__).resolve().parents[1]
    for name, expected in read(data / 'FILES_SHA256.json').items():
        require(sha(data / name) == expected, 'report data changed: ' + name)
    analysis = read(data / 'analysis.json'); verified = analysis['E3_recheck']
    ep = {(r['panel'], r['state']): r for r in load_csv(data / 'e3_state_endpoints.csv')}
    matched = {(r['panel'], r['state']): r for r in load_csv(data / 'matched_E1_E2_E3.csv')}
    ops = {(r['panel'], r['state'], r['operation']): r for r in load_csv(data / 'e3_operations.csv')}
    rescues = {(r['panel'], r['state'], int(r['third'])): r for r in load_csv(data / 'e3_rescues.csv')}
    tele = {(r['panel'], r['state'], r['order'], int(r['layer_added'])): r for r in load_csv(data / 'e3_telescopes.csv')}
    dist = {(r['panel'], r['state'], r['operation']): r for r in load_csv(data / 'e3_item_distributions.csv')}
    paired = {(r['panel'], r['operation']): r for r in load_csv(data / 'e3_C2_paired_temporal.csv')}
    supports = load_csv(data / 'e3_injection_support.csv')
    states = ['teacher'] + [f'{c}/step{s}' for c in CONDITIONS for s in (500, 2000, 10000)]
    final_states = ['teacher'] + [f'{c}/step10000' for c in CONDITIONS]
    folder = data.name

    def pair(source, state, field, scale=1):
        return ' / '.join('excluded' if source[p, state][field] == '' else f'{scale * float(source[p, state][field]):.6f}'
                          for p in ('discovery', 'confirmation'))

    def op_pair(state, label, field='delta_ce_nats'):
        return ' / '.join(f'{float(ops[p, state, label][field]):.6f}' for p in ('discovery', 'confirmation'))

    def layer_for(state, third): return (5, 17, 29)[third] if state == 'teacher' else (3, 11, 19)[third]

    trajectories = table(['State', 'All-layer deletion ΔCE D / C', 'Sink η=.10: three-probe mean ΔCE D / C',
                          'Sink η=.10: three-probe mean self-KL D / C'],
        [[s, pair(ep, s, 'all_delete_delta_ce_nats'), pair(ep, s, 'sink_eta01_coarse_mean_delta_ce_nats'),
          pair(ep, s, 'sink_eta01_coarse_mean_self_kl_nats')] for s in states])
    controls = table(['State', 'Layer', 'Sink ΔCE D / C', 'Random ΔCE D / C', 'Orthogonal ΔCE D / C', 'Non-sink ΔCE D / C'],
        [[s, layer_for(s, t)] + [op_pair(s, f'injection/{d}/layer{layer_for(s,t)}/eta0.1')
         for d in ('sink', 'random', 'orthogonal', 'non_sink')] for s in final_states for t in range(3)])
    doses = table(['C2/10000 layer', 'η', 'Sink ΔCE D / C', 'Random ΔCE D / C', 'Orthogonal ΔCE D / C', 'Non-sink ΔCE D / C'],
        [[l, f'{eta:g}'] + [op_pair('C2/step10000', f'injection/{d}/layer{l}/eta{eta:g}')
         for d in ('sink', 'random', 'orthogonal', 'non_sink')] for l in (3, 11, 19) for eta in (0, .01, .03, .1)])
    rescue_table = table(['State', 'Layer', 'Isolated deletion ΔCE D / C', 'All-layer + one donor ΔCE D / C',
                          'Conditional loss reduction D / C'],
        [[s, layer_for(s, t)] + [' / '.join(f'{float(rescues[p,s,t][field]):.6f}' for p in ('discovery', 'confirmation'))
         for field in ('single_delete_delta_ce_nats', 'conditional_rescue_delta_ce_nats', 'conditional_loss_reduction_nats')]
         for s in final_states for t in range(3)])
    telescope_table = table(['State', 'Layer', 'Forward increment D / C', 'Reverse increment D / C'],
        [[s, layer_for(s,t)] + [' / '.join(f'{float(tele[p,s,order,layer_for(s,t)]["increment_ce_nats"]):.6f}'
         for p in ('discovery', 'confirmation')) for order in ('forward', 'reverse')]
         for s in ('teacher', 'C2/step10000') for t in range(3)])
    distributions = table(['Panel', 'State / all-layer deletion', 'Mean ΔCE', 'Median', 'Q25', 'Q75', 'P90'],
        [[p, s] + [f'{float(dist[p,s,"all_delete/layer" + str(layer_for(s,0))]["delta_ce_"+k]):.6f}'
         for k in ('mean', 'median', 'q25', 'q75', 'p90')] for p in ('discovery','confirmation') for s in final_states])
    temporal = table(['Panel', 'C2 operation: 10,000 minus 500', 'Mean', 'Median', 'Q25', 'Q75', 'P90'],
        [[p,label] + [f'{float(paired[p,label][k]):.6f}' for k in ('mean','median','q25','q75','p90')]
         for p in ('discovery','confirmation') for label in ('all_delete/layer3',
            'injection/sink/layer3/eta0.1','injection/sink/layer11/eta0.1','injection/sink/layer19/eta0.1')])
    support_rows=[]
    for s in states:
        common=[]; excluded=[]
        for p in ('discovery','confirmation'):
            rows=[r for r in supports if r['state']==s and r['panel']==p and r['operation'].endswith('/eta0.1')]
            n=sum(int(r['eligible_positions']) for r in rows); c=sum(int(r['common_available_positions']) for r in rows)
            common.append(f'{c:,} / {n:,}'); excluded.append(str(n-c))
        support_rows.append([s, ' ; '.join(common), ' / '.join(excluded)])
    support_table=table(['State', 'Common / eligible: discovery ; confirmation', 'Excluded D / C'],support_rows)
    identity_table=table(['Identity', 'SHA-256 / commit'],[
        ['Original scientific executor','`f96c73061f9ef288c72f309e09c4c1f17a8a6701`'],
        ['Original E3','`5673eaefc30ff238ee338b0cff973032f6f2579e985592f0c837a04c6f24cf33`'],
        ['Nodi E3 successor','`36e202754ae7c2bca152f50f40ffe56bc520dc1a4ad05f4579721b307f56e03e`'],
        ['Original Adrita runtime','`df8b0b3258a9e9bd236e8a9d7281e95a68d3e5555733e76404fda338db5ada0b`'],
        ['Nodi runtime','`73c86bd2bd42383ef7648db5132d5dc1f644782440ec520cc0ced55ef1deb295`'],
        ['Final 32-bundle independent audit','`'+verified['final_audit_sha256']+'`']])
    e3_text=f'''# KD-SINK E3 scientific results

Prepared on 2026-10-11 (Asia/Dhaka) by Codex. **E3 is complete: 32 independently audited state/panel bundles, 9,600 item records and 1,920 pooled operation rows.** This is a quantitative analysis of the completed scientific measurements, extending the earlier [completion receipt](../E3_NODIPC_COMPLETION.md). It adds no model inference.

The main result is that simultaneous sink deletion becomes more damaging as these students train, while responses to fixed relative-norm injections depend on layer, direction and dose. On confirmation, all-layer ΔCE rises from 0.014148 to 0.401207 for C2, from 0.015926 to 0.368216 for C3, and from 0.010674 to 0.249293 for C6 between updates 500 and 10,000. The teacher's all-layer effect is larger, 1.560854 nats/target. Equal-norm sink-direction responses do not produce one teacher-like ranking: other directions can be as damaging or more damaging. Clean isolated-layer rescue succeeds; a single clean donor generally leaves substantial damage when all other native layers remain edited.

These are descriptive seed-0 results. Two disjoint text panels and two physical execution GPUs do not supply independent training replications. Read the [combined E1–E3 interpretation](MECHANISTIC_E1_E3_RESULTS_20261011.md) alongside this report.

## Design, completion and execution provenance

Frozen GPT-2-large is the teacher (36 layers, 20 heads, width 1,280). The GPT-2-medium students (24 layers, 16 heads, width 1,024) are the exact S1 seed-0 C1/C2/C3/C5/C6 states at updates 500, 2,000 and 10,000, originally trained from configuration initialization. E3 includes all 15 students and the static teacher on both panels.

| Panel | Bundles | Reused Adrita bundles | Fresh Nodi bundles | Item records |
|---|---:|---:|---:|---:|
| Discovery | 16 | 16 | 0 | 4,800 |
| Confirmation | 16 | 1 (teacher) | 15 | 4,800 |
| Total | 32 | 17 | 15 | 9,600 |

Each bundle has 300 frozen 128-token blocks: **38,400 input tokens and 38,100 scored next-token targets**. The campaign reuses 600 unique blocks, not 9,600 unique texts. Discovery spans 46 source documents; confirmation 32. Strict corpus reconstruction, document-ID/text-hash disjointness, exact tokens/masks/order and checkpoint bytes were recovered during migration. Within-panel blocks can share documents. Confirmation came from the frozen LM2000 region with prior S1 clean endpoint measurements; it is a holdout for these new interventions, not historically unseen data.

All 60 operations per item are measured: three probes × (four directions × four doses + isolated deletion + isolated rescue + all-layer deletion + conditional rescue). The all-layer deletion is recorded once per probe context but is the same native-all-layer intervention; this report counts its effect once per state and never treats its three copies as independent measurements.

Adrita's RTX 4080 SUPER UUID is `GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf`; Nodi's is `GPU-72b4b307-b613-c35e-ea32-53f4431de9ee`. Both follow-up environments use Python 3.12.3, PyTorch 2.10.0+cu128, Transformers 5.3.0, FP32 eager, cache disabled and TF32 disabled. Source-training hardware/precision in S1 identities is separate from follow-up execution. The successor records real device/runtime and relocated paths; it does not spoof the old GPU. Scientific-field diff is empty, and original envelopes/results are preserved.

The prospectively fixed 12-item live migration comparator showed exact measured agreement. All 32 full-300 E2 dependencies also passed exact deletion-effect/factor/geometry comparisons, covering **{analysis['E2_numeric_fields_exact']:,} numeric fields**. This supports the documented mixed-runtime analysis; it does not prove bitwise equality for every unrerun E3 tensor or an independent seed replication. Original and successor panel-file hashes differ because the relocated prepared artifact has new provenance; exact scientific item content/order is verified separately.

## Intervention definitions and denominators

Coarse probes are **student layers [3, 11, 19] and teacher layers [5, 17, 29]**, the approved lower-median representatives of native depth thirds. A probe edits one layer, not an entire third. Fine-layer selection and JVP analyses were not run. These coarse probes do not include the final student E2 peaks at layers 13 (C2), 12 (C3), or 15 (C6).

The injection grid is η = 0, .01, .03, .10. Requested per-query norm is `η × ||clean entering residual before ln_1||`; injection occurs at outgoing attention output before residual addition. The sink direction is the actual projected sink-deletion output delta. Controls are one deterministic Gaussian direction, its projection orthogonal to that sink direction, and projected non-sink value contrast `(v_key2 − v_key1) W_O`, with seed 20260927, keys [1,2], query_min 2 and floor 1e−8. The four directions use the same intersection of eligible supports. Equal relative norm within a state does not equate absolute norms, basis directions, model widths or functional dose across states.

Injection support is q=2–127 (126 positions/block before exclusions). Behavior scores q=0–126 (127 targets/block); q=127 has no within-block next-token target. Without exclusions, 125 scored positions/block are directly injected, with behavior still averaged over all 127 valid targets. Norm/support accounting therefore has 37,800 eligible positions per probe/bundle, not 38,100 prediction targets. Eta-zero outputs remain clean even where nonzero directions are undefined. Undefined support is reported, not assigned zero sensitivity.

ΔCE is edited-minus-clean loss in nats/target, self-KL is KL(clean || edited) over the full 50,257-token vocabulary, and flips are fractions of valid targets. Negative ΔCE means improved target loss. A three-probe mean averages **three separate interventions**; it is not their simultaneous effect or an architecture-independent circuit score. D / C means discovery / confirmation throughout.

## All-layer and equal-norm trajectories

{trajectories}

All-layer damage increases at every observed interval in all five student conditions on both panels. C2/C3/C6 finish substantially above C1/C5, but all remain below the teacher's simultaneous effect. In contrast, sink-direction injection responses need not increase monotonically or follow the same ordering. C2's three-probe mean sink-direction ΔCE at η=.10 is {pair(ep,'C2/step500','sink_eta01_coarse_mean_delta_ce_nats')} at update 500 and {pair(ep,'C2/step10000','sink_eta01_coarse_mean_delta_ce_nats')} at 10,000. A near-zero or negative signed mean is not absence of output sensitivity; self-KL and item heterogeneity remain visible.

![Completed E1–E3 trajectories]({folder}/combined_trajectories.png)

*Teacher references are static; student lines connect measured checkpoints. E2 peaks are retrospective maxima. E3 injection averages are summaries of three separate coarse-layer edits.*

## Every coarse-layer control at update 10,000

The following table fixes η=.10 for a readable endpoint comparison and retains every prescribed direction and layer. Other doses, intermediate checkpoints, geometry and self-KL are in the full operation export.

{controls}

At C3 layer 11 on confirmation, sink and random ΔCE are 0.002909 and 0.002830; non-sink is 0.007508. At C6 layer 11, sink is 0.003057 and random/orthogonal are about 0.003119. At teacher layer 29, random ΔCE is 0.004895 versus sink 0.007300, while non-sink is 0.007733. These are measured directional differences, not practical equivalence or significance tests. Sink direction is not universally the most damaging equal-norm direction.

### Full approved C2 endpoint dose curves

{doses}

The plots below show all approved doses and directions for the teacher and the registered C2 endpoint on confirmation; numerical discovery counterparts remain in the table/CSV. Small η responses can be negative and curves need not be linear. No slope extrapolation, fitted onset, or response-based dose selection is introduced.

![Equal-norm dose responses]({folder}/e3_dose_response.png)

## Clean-output rescue and conditional restoration

Isolated deletion plus the clean donor attention output restores clean outputs. The largest absolute pooled isolated-rescue ΔCE is **{analysis['max_single_rescue_absolute_delta_ce']:.12g}**; the largest eta-zero absolute pooled ΔCE is **{analysis['max_eta0_absolute_delta_ce']:.12g}**. Their original live logit checks also passed. Zero scalar loss change alone is not the complete tensor-level proof; the qualified runner supplies that proof.

`All-layer + one donor` restores the clean attention output at the chosen probe while every other native layer remains sink-deleted. `Conditional loss reduction` is all-layer ΔCE minus this conditional ΔCE. It is a signed difference in nats/target, can be negative, and is neither a mediated fraction nor an additive contribution.

{rescue_table}

For C2 on confirmation, restoring layer 19 reduces all-layer ΔCE by 0.042687, but leaves 0.358520 rather than returning to clean. Restoring layer 11 reduces it by 0.011049 and leaves 0.390159. For the teacher, restoring layer 17 reduces damage by 0.357011 and still leaves 1.203843. The same layer's isolated deletion and its restoration in an already edited network answer different conditional questions. Smaller rescue effects do not establish that a layer is unnecessary, and larger ones do not establish unique mediation.

## Ordered telescopes and nonlinear dependence

Every bundle retains complete ascending and descending native-layer orders. The sum of ordered increments equals the all-layer endpoint within the original numerical gates. Individual increments depend on the preceding edited layers and may be negative. The table below uses only the predefined coarse probes, with complete native-layer accounts in `e3_telescopes.csv`.

{telescope_table}

The common endpoint is invariant to the final ordering; the increment assigned to a layer is not. For C2/10,000 confirmation, layer 11 contributes 0.006638 in forward order and 0.013998 in reverse order. These are order-specific accounts, not independent circuit contributions. FP32 behavior ΔCE and the telescope's recorded CE account can differ at small numerical scale; their declared closure checks remain unchanged.

## Item distributions and the registered temporal contrast

All-layer deletion distributions retain signed effects on the same 300 items per state/panel. Quartiles and p90 describe heterogeneous items; they are not confidence intervals or independent-document uncertainty.

{distributions}

C2's prespecified temporal comparison pairs each update-500 item with the identical update-10,000 item. Its all-layer contrast appears once; normalized sink-direction contrasts are shown for all three fixed probes at η=.10.

{temporal}

These changes mix the evolving network's response with its evolving checkpoint-local directions/reference norms. They do not isolate a single invariant vector through training.

## Exact support and numerical integrity

The table sums query-layer observations across the three probes at η=.10. It includes q=127 and is deliberately separate from prediction-target counts. Counts for every layer/dose and direction-specific exclusions remain in `e3_injection_support.csv`.

{support_table}

Exclusions at η=.10 total **{analysis['support_excluded_positions']:,} query-layer observations**: early C1 and update-2,000 C6 have incomplete common support; all other state/panel rows have full support. Controls share the same intersection within each item, but support need not be identical across checkpoints. These exclusions constrain cross-state comparisons and are not zero-effect observations.

Full raw reporting verification rehashed **{verified['raw_E3_files_rehashed']:,} item files**, recomputed **{verified['operation_pools_recomputed']:,} operation pools** and checked {verified['pooled_numeric_checks']:,} pooled scalars. Maximum absolute raw-to-summary discrepancy was `{verified['max_absolute_repool_discrepancy']:.17g}` across those heterogeneous scalar units; comparison also checks the original units and denominators. All four control records share exactly matching support metadata in every item/layer/dose. No unavailable item-operation was silently dropped: count **{verified['unavailable_item_operations']}**. This does not imply that every direction had every eligible query available.

| Original audit diagnostic | Maximum recorded absolute error |
|---|---:|
| E2 componentwise output/logit parity | {verified['max_parity_absolute_error']:.17g} |
| Delivered injection norm | {verified['max_injection_norm_error']:.17g} |
| Double full-vocabulary loss geometry | {verified['max_double_geometry_error']:.17g} |

Original acceptance gates remain parity abs 1e−3 + rel 1e−4 × |reference|, norm abs 1e−6 + rel 1e−6 × requested norm, and double geometry abs 1e−10 with zero relative slack. These are numerical correctness checks, not scientific equivalence margins. Historical norm-audit failures and corrected population bookkeeping are preserved in the migration record. Per-query clean residual tensors were not serialized, so their historical reconstruction remains unavailable; original live norm checks, audited scalar support accounting and fresh fixed-item qualification are the disclosed evidence. No threshold was relaxed to obtain a desired scientific outcome.

{identity_table}

The immutable archive and imported Adrita attempts remain external. New results live in `C:/KD-E3-Nodi-20261010/campaign_nodipc`; final provenance is in `final_completion/ALL_PHASE_PORTABLE_JOIN.json`. This report reuses existing runner/full independent semantic audits, freshly checks every E3 raw item byte, repools scalar metrics and verifies support/telescope metadata. It does not repeat model inference or reconstruct unsaved activation tensors.

## Interpretation limits and reproducible exports

E3 separates direction-specific downstream response from natural deletion magnitude and from the E1 parameter routes. Equal residual-normalized injections do not supply matched absolute activation dose across architectures. One fixed Gaussian control is a direction control, not a random-direction ensemble. Coarse probes cannot resolve unsampled sensitive layers; all-layer deletion changes many layers together. Rescue and telescopes demonstrate conditional/nonlinear behavior without identifying a unique causal mediator. No across-seed reproducibility, p-values, confidence intervals, equivalence decisions, composite inheritance score or unmeasured later-training convergence is claimed.

All 1,920 unrounded [operation rows]({folder}/e3_operations.csv), [support counts]({folder}/e3_injection_support.csv), [state endpoints]({folder}/e3_state_endpoints.csv), [rescues]({folder}/e3_rescues.csv), [ordered telescopes]({folder}/e3_telescopes.csv), [item distributions]({folder}/e3_item_distributions.csv), and [all C2 temporal contrasts]({folder}/e3_C2_paired_temporal.csv) are included. [Analysis/provenance]({folder}/analysis.json) and [file hashes]({folder}/FILES_SHA256.json) bind the compact exports; raw item/target records, datasets, weights and bulk logs remain outside Git. SVG versions of both figures are included for export.

The [fresh reporting verification receipt](mechanistic_e1_e3_verification_20261011.json) records source/report/code hashes, exact commands, 19 passing CPU tests and verification scope.

```powershell
.venv/Scripts/python.exe scripts/summarize_mechanistic_e1_e3.py --root C:/KD-E3-Nodi-20261010 --output reports/{folder}
.venv/Scripts/python.exe scripts/render_mechanistic_e1_e3_reports.py --data-directory reports/{folder}
```

Read the [execution contract](../design/e6a_v2/MECHANISTIC_E1_E3_EXECUTION.md), [E3 specification](../design/e6a_v2/studies/MECHANISTIC_E3_PROTOCOL_v1.md), original approved envelope and versioned Nodi successor with their later approval records. Historical draft-status prose does not replace those actual approved identities.
'''
    combined_table = table(['Final state', 'E1 clean sink D / C', 'E1 Q removal % D / C', 'E2 peak isolated ΔCE D / C',
                           'E3 all-layer ΔCE D / C', 'E3 sink η=.10 mean ΔCE D / C'],
        [[s, pair(matched,s,'E1_clean_sink'),pair(matched,s,'E1_q_bias_removed_fraction',100),
          pair(matched,s,'E2_observed_peak_delta_ce_nats'),pair(ep,s,'all_delete_delta_ce_nats'),
          pair(ep,s,'sink_eta01_coarse_mean_delta_ce_nats')] for s in final_states])
    nonlinear_rows=[]
    for s in final_states:
        n=36 if s=='teacher' else 24
        iso=[n*float(matched[p,s]['E2_native_layer_mean_delta_ce_nats']) for p in ('discovery','confirmation')]
        all_delta=[float(ep[p,s]['all_delete_delta_ce_nats']) for p in ('discovery','confirmation')]
        nonlinear_rows.append([s,' / '.join(f'{v:.6f}' for v in iso),' / '.join(f'{v:.6f}' for v in all_delta),
                               ' / '.join(f'{a-i:.6f}' for a,i in zip(all_delta,iso))])
    nonlinear_table=table(['Final state','Sum of isolated E2 ΔCE D / C','E3 all-layer ΔCE D / C','All-layer minus isolated sum D / C'],nonlinear_rows)
    full_join_table=table(['State','E1 clean sink D / C','E1 Q removal % D / C','E2 peak isolated ΔCE D / C','E3 all-layer ΔCE D / C'],
        [[s,pair(matched,s,'E1_clean_sink'),pair(matched,s,'E1_q_bias_removed_fraction',100),
          pair(matched,s,'E2_observed_peak_delta_ce_nats'),pair(ep,s,'all_delete_delta_ce_nats')] for s in states])
    combined=f'''# KD-SINK combined E1–E3 scientific report

Prepared on 2026-10-11 (Asia/Dhaka) by Codex from **92 independently valid bundles: E1 28, E2 32, E3 32**. All phases are complete. This report connects the [detailed E1/E2 report](MECHANISTIC_E1_E2_RESULTS_20261010.md) with the [new E3 report](MECHANISTIC_E3_RESULTS_20261011.md), using exact matched checkpoint weights and panel items. It introduces no new scientific inference.

**Sink pattern, the tested parameter routes and causal response do not transfer together as one uniform package.** C2/C3/C6 acquire large sink patterns early, while Q-bias/EPE route responses remain much weaker than the teacher's. Their isolated and simultaneous deletion sensitivity grows later. E3 adds that normalized downstream response depends on direction and depth, and that simultaneous deletion is strongly nonlinear. The teacher has greater simultaneous all-layer damage than any final student even though final C2/C3 have larger observed isolated-layer peaks. A single statement such as “sinks transfer without function” or “students inherit the teacher mechanism” would omit important components of these measurements.

## Shared design and evidence

The teacher is frozen GPT-2-large (36 layers/20 heads); students are independently trained-from-configuration GPT-2-medium (24 layers/16 heads), exact S1 seed-0 checkpoints. C1 is behavioral CE/logit-KD, C2 full cosine-soft attention JSD, C3 calibrated head-mean probability MSE, C5 conditional non-sink attention JSD and C6 sink-versus-rest JSD. The adaptations and fixed layer map are retained; no head/circuit homology is inferred. C0/C4 and additional seeds are outside this follow-up grid.

| Phase | Scientific question | Bundles | Item records | Grid |
|---|---|---:|---:|---|
| E1 | Q/K parameter routes, sink dose and broad positional diagnostics | 28 | 8,400 | 13 students + teacher, two panels |
| E2 | Validated isolated sink deletion and local output factors | 32 | 9,600 | 15 students + teacher, two panels |
| E3 | Equal-norm directional response, rescue and simultaneous/ordered deletion | 32 | 9,600 | 15 students + teacher, two panels |
| Total | Phase-specific measurements | 92 | 27,600 | 600 unique blocks reused |

Each bundle has 300 blocks, 38,400 input tokens and **38,100 valid shifted prediction targets**. All behavior changes are token weighted in nats/target; sink removals are item-weighted guarded fractions; factors use their declared query supports. Summaries do not turn repeated states/interventions into extra unique observations. The 300-item panels have disjoint source-document IDs and normalized text hashes, but blocks within a panel can share documents. Confirmation is intervention-held-out rather than wholly unseen historical data because its frozen LM2000 region had S1 endpoint measurements.

E1 intentionally excludes C1/C5 at update 2,000. E2/E3 include them, and the joint CSV explicitly leaves those E1 cells blank. The historical E1/E2 report's “E3 running” statement describes its 2026-10-10 preparation snapshot; it is preserved as a historical report, and this document gives the completed status.

All E1/E2 measurements and 17 E3 bundles are Adrita outputs. The remaining 15 E3 confirmation student bundles ran on Nodi under the researcher-authorized successor. Each bundle retains its own sealed runtime/protocol; no original E1/E2 execution is relabeled as Nodi. Exact fixed-item migration comparisons, all 32 matched E2 dependencies and an empty scientific-field diff support the explicit mixed-runtime join. The [migration record](../E3_NODIPC_MIGRATION_FEASIBILITY.md) describes qualification, path relocations and historical audit corrections. Two physical GPUs are not replicated training seeds.

## Matched endpoint comparison

D / C denotes discovery / confirmation. E2 `peak` is the largest observed isolated native-layer ΔCE and can select a different layer per model/panel. E3 injection mean describes separate probes at predefined coarse thirds. They are different estimands, with different layer coverage.

{combined_table}

The teacher's Q-bias removal is roughly 24%, versus about 0.72–3.55% in final students. C3 has almost teacher-sized mean sink mass but only about 0.93% Q-bias removal. Nevertheless, final C3/C2 isolated-deletion peaks exceed the teacher's on both panels. E3 all-layer deletion gives the opposite teacher/student magnitude ordering, with the teacher substantially larger. These statements can hold together because isolated maxima and simultaneous interventions answer different questions.

![Separate pattern, route and causal trajectories]({folder}/combined_trajectories.png)

*Full registered checkpoint trajectories; confirmation solid, discovery dashed, static teacher black. E1 lines for C1/C5 connect the observed endpoints without imputing update 2,000.*

## Pattern and route evidence from E1

C2/C3/C6 have substantial key-0 attention by update 500. C3 moves near the teacher's native mean amplitude at update 10,000; C2's pattern is nonmonotonic while its tested Q/K dependence grows. C1/C5 retain small sinks. Q-bias removal and layer-0 EPE transport distinguish the teacher from high-sink students, while top-three K-input dependence grows through training. Coordinates are chosen separately per model and are not homologous.

Equal parameter α is not equal sink or activation dose. In each panel, only **3/273** approved E1 dose comparisons have unique observed overlap: top-three K input at 10% relative sink removal for final C1/C3/C5. The other 270 are dose-unmatched; no supported Q-bias or 25%/50% matches exist. Missing dose overlap cannot establish a null or equivalence. Broad position-0→1 interventions strongly affect both pattern and behavior, but do not isolate one Q/K mechanism.

The detailed E1/E2 report preserves every α, teacher native/mapped scope, all five random-coordinate controls and signed effects. C3's supported K-top3 10% comparisons have more ΔCE damage than the mapped teacher on both panels; C1/C5 have less. Relative removal still does not equate absolute sink mass, residual bases or broader activation dose. The evidence supports component-specific route differences, not a universal inheritance/absence verdict.

## Natural deletion, local factors and downstream response

E2 measures one native layer at a time and verifies its immediate attention-output delta against actual deletion and clean-delta injection. Sink mass multiplies a conditional non-sink-versus-sink value contrast; projection and cross-head cancellation also matter. The teacher's mean relative local output change is larger and its projected heads cancel more strongly than final students. These architecture-local diagnostic averages do not by themselves predict which downstream intervention produces most loss damage.

E3's normalized sink direction fixes a fraction of each clean entering residual norm. It separates response to that local direction from the naturally delivered deletion magnitude, while preserving direction-specific controls. It does not hold the same absolute vector or activation norm constant across models/checkpoints. A low-sink checkpoint can still respond to a normalized direction; a high-sink checkpoint can show comparable or greater response to a non-sink control. Random/orthogonal/non-sink responses therefore remain part of the result rather than disposable nulls.

The predefined E3 coarse probes are teacher [5,17,29] and student [3,11,19]. They omit the final E2 student peaks at C2 layer 13, C3 layer 12 and C6 layer 15. E3's coarse injection measurements cannot explain or refute sensitivity at those unprobed E2 peaks. The fine-layer extension was not run. Across matched coarse layers, original E2 and E3 isolated-deletion behavior/geometry are exactly equal; no phase drift is concealed.

Common direction support excludes 7,803 query-layer observations at η=.10 across early C1 and update-2,000 C6; every other state/panel has full eligible support. Every direction uses the same support within a given item, while cross-checkpoint supports can differ. The E3 support tables retain those counts; excluded queries are not interpreted as zero sensitivity.

## Simultaneous deletion is not the sum of isolated effects

The next table compares all-layer E3 ΔCE with the numerical sum of all separate E2 isolated-layer ΔCE values. The difference is an observed nonadditivity account, not a “synergy percentage,” mediation fraction or independent interaction estimate.

{nonlinear_table}

Every final state's simultaneous effect exceeds its sum of isolated effects on both panels. The teacher's confirmation isolated sum is much smaller than its 1.560854 simultaneous ΔCE. Thus averaging or summing isolated effects would understate the measured all-layer response. Complete forward/reverse telescopes reach the same final endpoint, but assign different increments to layers depending on prior edits; those increments are conditional/order-dependent.

Clean-output isolated rescue returns to clean under original logit gates. In an all-layer-deleted network, restoring a single coarse clean donor leaves substantial damage. C2 confirmation layer-19 restoration leaves 0.358520 ΔCE from an all-layer 0.401207; teacher layer-17 restoration leaves 1.203843 from 1.560854. Those conditional rescue differences are not unique layer contributions. See the E3 report for every endpoint rescue and both full layer-order accounts.

## Full checkpoint context

{full_join_table}

The primary temporal contrast is C2 500→10,000, with 2,000 descriptive. E3 all-layer damage increases at all observed student intervals on both panels; the E1 pattern/Q/K trajectories and E3 equal-norm responses have their own shapes. No global convergence classification is imposed on this multi-component evidence. C1/C5 E1 update-2,000 entries are protocol-excluded, not failed measurements.

Students also remain worse language models than the teacher: confirmation endpoint clean CE is 3.159942 for the teacher and roughly 4.148–4.174 for these students. Absolute ΔCE, relative PPL changes, full-vocabulary self-KL, flips and signed target-loss geometry are distinct. Similar clean student performance does not equate intervention responses, and architecture/clean-performance differences constrain teacher/student causal comparisons.

## Conclusions supported within this campaign

1. Pattern resemblance does not establish the teacher's tested parameter routes: strong early student sinks coexist with much weaker Q-bias/EPE responses.
2. Student causal sensitivity develops and can exceed the teacher for selected isolated layers, while simultaneous all-layer damage remains larger in the teacher.
3. Equal relative-norm responses vary by direction/depth/dose. The sink direction is not uniformly the most damaging control, and cross-state vector identity is not fixed.
4. Rescue and ordered telescopes show conditional/nonlinear response. They do not identify a unique mediator or justify adding isolated effects into a simultaneous causal decomposition.
5. Discovery/confirmation broadly agree on these descriptive patterns, while effect magnitudes and some layer maxima differ. Confirmation is not an independent training-seed replication.

One seed, two related corpus panels, finite coarse probes and one deterministic random direction limit generalization. There are no inferential statistics, confidence intervals, practical equivalence margins, inferred onset thresholds, mediation percentages, composite scores, JVP/fine extensions or unmeasured late-training claims. Ten thousand optimizer updates is the observed horizon, not proof of convergence.

## Verification, artifacts and reproduction

The original E1/E2 report, verification receipt, exporter and every companion hash were freshly checked. All four numeric E1/E2 CSVs regenerated **byte-identically** from the 60 selected sealed source summaries and SHA-bound dual audits. Their arithmetic, state grid, source hashes, archived joins and item pairing pass. The prior focused 1,800-item independent E1/E2 check remains explicitly reused rather than presented as a new raw E1/E2 audit.

For E3, this reporting pass rehashed all **9,600 raw item files**, independently repooled all **1,920 operation cells**, verified every direction's common-support metadata and checked raw telescope closure. Full original tensor gates and independent semantic audits are reused; unsaved activation tensors are not reconstructed. All 92 summaries match the final portable join, every source weight and panel item is paired correctly, and original/successor runtime identities remain explicit. CPU report tests and a receipt bind the new report code/data; the documents are audited scientific reporting rather than a new experiment.

The original-versus-successor seals, recorded numerical maxima and historical audit limitations are listed in the [E3 report](MECHANISTIC_E3_RESULTS_20261011.md). Raw sources remain under `C:/KD-E3-Nodi-20261010/imported_adrita/campaign` and `campaign_nodipc`; reconstruction/receipt history is under `reconstruction01`. The immutable ZIP, interrupted attempts and original approved protocols remain preserved.

- [Combined unrounded state table]({folder}/matched_E1_E2_E3.csv): all 32 E2/E3 state-panel pairs, explicit E1 availability, source weights and E3 origin.
- [Analysis and fresh verification]({folder}/analysis.json): all source identities, audit maxima and disclosure of reused versus fresh checks.
- [Reporting verification receipt](mechanistic_e1_e3_verification_20261011.json): exact commands, test evidence and both report hashes.
- [E3 operation/support/rescue/telescope/distribution companions]({folder}/FILES_SHA256.json): complete numeric export inventory.
- [Original E1/E2 companion inventory](mechanistic_e1_e2_results_20261010/FILES_SHA256.json): original 2,180 operations, 1,584 factor rows and all E1 dose statuses.
- [Exportable combined figure]({folder}/combined_trajectories.svg) and [E3 dose figure]({folder}/e3_dose_response.svg).

Reproduction is CPU-only using the two reporting commands in the E3 report. Existing experimental outputs, datasets, weights and bulk logs are not committed or changed. Reporting does not launch training, E4/E5 or Stage09.
'''
    for name, text in [('MECHANISTIC_E3_RESULTS_20261011.md', e3_text), ('MECHANISTIC_E1_E3_RESULTS_20261011.md', combined)]:
        (repo / 'reports' / name).write_text(text, encoding='utf-8', newline='\n')
    print('PASS: E3 scientific report and combined E1-E3 report rendered from sealed numerical exports')


if __name__ == '__main__':
    main()
