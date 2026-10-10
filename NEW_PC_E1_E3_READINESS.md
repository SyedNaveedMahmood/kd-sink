Current scope update (2026-10-10): the researcher has now authorized E3 migration
and scientific continuation. Independent feasibility is READY_FOR_PORTABLE_E3_CONTINUATION;
all original E1/E2 evidence is verified and17 E3 bundles can be reused. See
[E3_NODIPC_MIGRATION_FEASIBILITY.md](E3_NODIPC_MIGRATION_FEASIBILITY.md).
The setup-only approval block recorded below is historical.

# New PC E1–E3 readiness — 2026-10-10

**BLOCKED for scientific execution: researcher approval of new device/runtime envelopes is required. Engineering setup is READY. Missing required artifacts: none; no further transfer or downloads are needed.** No scientific experiments, training, E4/E5 or Stage09 were launched.

Repository: `C:\Users\user3\kd-sink`, tracking `origin/mechanistic-e0` at setup commit `b134a5ca30e858d2bec7bc4b9cc4cd21eb3104a6`. Initial clean main branch is preserved. All 46 scientific code/dependency pins match validated source `f96c73061f9ef288c72f309e09c4c1f17a8a6701`; four specification pins also match. Checkout CRLF was converted to the exact approved LF bytes, with no logic changes. `Upstream/` remains untouched.

Transfer folder discovered as `C:\Users\user3\kd-sink files`. Verified inputs are outside Git under `C:\KD-SINK-new-PC-E1-E3`:

| Required input | Verified local location / original source |
|---|---|
| Frozen GPT-2-large + original tokenizer/student config | `artifacts\KD-SINK-stage06-production\{teacher,tokenizer,student-config}`; Stage06 ZIP original root `F:\KD-SINK-stage06-production` |
| Original OWT corpus/panels | `artifacts\KD-SINK-stage06-production\{corpus,panels}`; same Stage06 ZIP and original artifact inventory |
| S1 seed0 C1/C2/C3/C5/C6 at 500/2000/10000 | `runs\s1-cX-seed0-rtx4080super\checkpoints\{weights-000500,weights-002000,final-010000}`; original verified root `D:\KD-SINK-central\runs` |
| Original approved E1/E2/E3 envelopes | Repository `protocols\mechanistic_e1_e3_approved_20261009\E*.approved.json`; original researcher approval preserved |
| Frozen discovery300 + confirmation300, selection receipt | `transfer_snapshot\preparation\attempt01\P1_panel_retry02`; original `D:\KD-SINK-central\analysis\mechanistic_preparation_20261008\attempt01\P1_panel_retry02` |

All 27 Stage06 files passed final SHA/CRC verification against the original archive and applicable manifests. Stage06 outer SHA: `220e428b54ba9c1e175826d3bcc14ac04ed0e5e2dcf9a59990d546ce32299d11`. Mechanistic transfer outer SHA: `4000dc81102b809020beeda491b20cb3b7dc309c0bfb787290e3fc50a25f8378`; original member inventory and all 60 selected members passed verification.

Checkpoint inference subsets passed ZIP CRC, original transfer SHA tables, approved weight/manifest hashes and manifest/COMPLETE/S1 identity gates. Full S1 archives were preserved; outer full-run hashes were not redundantly repeated. C1 comes from `_GPU-f6ff547d.zip`; other conditions come from their corresponding `S1_CX_seed0_*.zip`. Optimizer `state.pt` remains archived, so the extracted inference subsets must not be used for training resume.

Host `DESKTOP-POT9NL1`: RTX 4080 SUPER, UUID **GPU-72b4b307-b613-c35e-ea32-53f4431de9ee**, VRAM **16,376 MiB / 17,170,956,288 bytes**, driver **591.86**. Driver-supported CUDA is **13.1**; the pinned PyTorch runtime is **CUDA 12.8**. `.venv` uses Python **3.12.3**, torch **2.10.0+cu128**, transformers **5.3.0**. Offline `uv sync --locked` and dependency compatibility checks passed.

Validation passed:

- Focused CPU/environment tests: **56 passed, zero failed/skipped**. Local tokenizer load/encode/decode passed.
- E1/E2/E3 128-token synthetic GPU smokes: complete, reverified, and independently audited with the stdlib auditor.
- Real GPT-2-large plus trained GPT-2-medium C2/step10000: all six phase/model synthetic bundles qualified and independently audited. All 15 student source identities/hashes were checked. Strict FP32/eager/cache-off/TF32-off, batch1/chunk16 and original numerical gates retained. Qualification took **124.71 seconds**, peak reserved VRAM **5.637 GiB**, minimum free VRAM **9.439 GiB**, required headroom **1.599 GiB**, maximum **57°C**, minimum available RAM **45.074 GiB**.

The full OWT packing reconstruction was not redundantly repeated during setup. Original corpus and panel bytes are hash-verified, panel membership/tokens are unchanged, and document-disjoint splits were checked. Scientific admission still performs its complete reconstruction. Setup helper failures and their corrections are retained externally; no scientific checks were weakened.

The old Adrita UUID is `GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf`. All three original approved locks correctly reject this PC before model loading. New qualification-bound **drafts** are in `candidates_for_researcher_approval`; the sealed prospective amendment is `PROSPECTIVE_DEVICE_RUNTIME_AMENDMENT.json` (digest `2286253614ce1991eda30f35e3b7d62a9c22cd7dab138092c75d44a550bc487f`). It requires researcher approval of the new fixed UUID/runtime, local references, qualification receipts, output lineage and relocated panel hash. Original frozen panel SHA `509a90c6039cd90d7d4f6986ac7fe3b75468609073b4808baa8b5bae5294c7f7` is preserved; `panel\panel.json` changes only embedded source paths, SHA `e624cf4c30c67704fb65fbe02c8d9aa8213bc5995dff256b10c9bb01676bd2f7`.

After approval, issue fresh successors at `C:\KD-SINK-new-PC-E1-E3\approved\E*.approved.json`, preserve the drafts, and obtain explicit operator launch scope. Keep new-runtime outputs separate from the transferred Adrita campaign; phase joins reject mixed locks/runtime. E3 requires independently audited matching E2 state/panel evidence. Use the approved phase order and complete grids: E1 14 states including teacher, E2/E3 16 each, both panels, totaling 92 separately launched bundles.

These commands launch the first teacher/discovery bundle of each phase **only after those approvals and prerequisites**. Set `$E1ApprovedSHA`, `$E2ApprovedSHA`, `$E3ApprovedSHA` to the externally issued researcher-approved successor digests; draft digests cannot authorize runs. Each command launches one explicit seed/state/panel and needs a fresh output directory.

```powershell
Set-Location 'C:\Users\user3\kd-sink'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:HF_HOME='C:\KD-SINK-new-PC-E1-E3\artifacts\KD-SINK-stage06-production\hf-home'
$env:OMP_NUM_THREADS='2'

& .\.venv\Scripts\python.exe scripts/run_mechanistic.py scientific `
  --phase E1 --seed 0 --state teacher --panel discovery --device cuda:0 --disable-tf32 `
  --lock C:\KD-SINK-new-PC-E1-E3\approved\E1.approved.json --approved-sha256 $E1ApprovedSHA `
  --output C:\KD-SINK-new-PC-E1-E3\scientific\E1_discovery_teacher_attempt01

& .\.venv\Scripts\python.exe scripts/run_mechanistic.py scientific `
  --phase E2 --seed 0 --state teacher --panel discovery --device cuda:0 --disable-tf32 `
  --lock C:\KD-SINK-new-PC-E1-E3\approved\E2.approved.json --approved-sha256 $E2ApprovedSHA `
  --output C:\KD-SINK-new-PC-E1-E3\scientific\E2_discovery_teacher_attempt01

& .\.venv\Scripts\python.exe scripts/run_mechanistic.py scientific `
  --phase E3 --seed 0 --state teacher --panel discovery --device cuda:0 --disable-tf32 `
  --lock C:\KD-SINK-new-PC-E1-E3\approved\E3.approved.json --approved-sha256 $E3ApprovedSHA `
  --output C:\KD-SINK-new-PC-E1-E3\scientific\E3_discovery_teacher_attempt01
```

Exact test/qualification commands, source locations, local path configuration, amendment drafts, manifests and receipts are under `C:\KD-SINK-new-PC-E1-E3` and in [the small readiness evidence](reports/new_pc_e1_e3_readiness_20261010.json). First next action: researcher reviews the three qualified drafts and approves successors; scientific launch remains prohibited.
