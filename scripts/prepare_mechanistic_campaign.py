"""Seal only the exact researcher-approved 2026-10-09 candidates, read-only inputs."""
import argparse
import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from sinklab.mechanistic_admission import runtime_identity, _source, E1_GRID, E2_GRID
from sinklab.mechanistic_run import read_json, sha256_file, write_json
from sinklab.provenance import verify_envelope, seal_payload, payload_digest
from sinklab.followup_policy import D24_SHA256

APPROVED = {
    'E1': '424739bcf88f2949023dc5b369c234715aae1b96fa8e5193cd0240f61336530b',
    'E2': 'd6b49094f43a7814b3d60217f31dcfac51c93df320984d12469ce258fb31df52',
    'E3': '3001fd3aad8f6a634f949b41b217f17972610a139fbd388bbb4c7dfb712252eb',
}
PANEL = '509a90c6039cd90d7d4f6986ac7fe3b75468609073b4808baa8b5bae5294c7f7'
SOURCE = 'f96c73061f9ef288c72f309e09c4c1f17a8a6701'


def check(value, reason):
    if not value:
        raise ValueError(reason)


def seal(repo, candidates, output, authorization):
    check(not output.exists(), 'new approval directory required')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    # Approval source is the reviewed preparation milestone, independent of later
    # operational-only commits. Every execution-critical file remains pinned.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = runtime_identity(repo, 'cuda:0')
    documents = {}
    sources = {}
    for phase, digest in APPROVED.items():
        candidate = candidates / f'{phase}_v2.candidate.json'
        payload, actual = verify_envelope(read_json(candidate))
        check(actual == digest, f'{phase} is not the researcher-approved candidate')
        check(payload['runtime'] == runtime, 'qualified execution runtime changed')
        check(payload['status'] == 'draft' and not payload['production_ready'], 'unexpected predecessor status')
        check(payload['grid'] == ['teacher', *(E1_GRID if phase == 'E1' else E2_GRID)], 'grid changed')
        check(payload['seed'] == 0 and payload['d24_sha256'] == D24_SHA256, 'D24/seed changed')
        check(sha256_file(payload['panel']['path']) == payload['panel']['sha256'] == PANEL, 'panel changed')
        for name, sha in payload['specification_files_sha256'].items():
            check(sha256_file(repo / name) == sha, 'scientific specification changed')
        q = payload['qualification']
        check(sha256_file(q['path']) == q['sha256'], 'qualification changed')
        measured = read_json(q['path'])
        check(measured['runtime'] == runtime and measured['settings_sha256'] == payload_digest(payload['settings']), 'qualification/settings mismatch')
        check(measured['parity_passed'] and measured['headroom_passed'], 'qualification incomplete')
        for state, reference in payload['sources'].items():
            if state not in sources:
                sources[state] = _source(reference, state)
        approved = copy.deepcopy(payload)
        approved.update(status='approved', production_ready=True, source_commit=SOURCE)
        approved['approval'] = {
            'authority': 'researcher', 'record': 'Explicit CODEX TASK — Unattended E1–E3 Scientific Campaign with Luna Subagent Handoff, 2026-10-09; exact v2 candidates and seed0 grids authorized',
            'authorization_file_sha256': sha256_file(authorization),
            'candidate_predecessor_sha256': actual, 'candidate_file_sha256': sha256_file(candidate),
            'approved_utc': datetime.now(timezone.utc).isoformat(),
            'execution_source_commit': SOURCE, 'runtime_sha256': payload_digest(runtime),
            'sealing_checkout_commit': source,
        }
        approved['resolved_researcher_decisions'] = approved.pop('pending_researcher_decisions')
        documents[phase] = seal_payload(approved)
    output.mkdir(parents=True)
    for phase, document in documents.items():
        write_json(output / f'{phase}.approved.json', document)
    write_json(output / 'APPROVAL_RECEIPT.json', {
        'status': 'APPROVED_EXACT_V2', 'source_commit': SOURCE,
        'runtime_sha256': payload_digest(runtime), 'code_sha256': runtime['code_sha256'],
        'candidate_predecessors': APPROVED, 'panel_sha256': PANEL,
        'approved_envelope_sha256': {p: d['sha256'] for p, d in documents.items()},
        'sources_read_only_verified': sources, 'authorization_sha256': sha256_file(authorization),
    })
    print(json.dumps({p: d['sha256'] for p, d in documents.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args()
    seal(Path(__file__).resolve().parents[1], args.candidates, args.output, args.authorization)
