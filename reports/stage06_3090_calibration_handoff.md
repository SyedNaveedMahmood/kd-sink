# Stage 06 RTX 3090 calibration handoff

Historical handoff: the original tested source milestone was `aef95f328fe052c0b0e79502823b05b66d8dc70e` on the RTX 4080 SUPER, followed by record-only attribution commits. The RTX 3090 checkout began at `9dda2efef8f5e2111cf0745f37a68f39c1cf6576` on the approved device (`GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e`). Its exact locked environment was then sealed in `protocols/s1_environment_3090_exact_v1.json`, with all 101 lock-managed distributions equal to the 4080 evidence. At that handoff the external `F:\KD-SINK-stage06-production` tree was not mounted, so calibration and final production locks remained blocked. The original invocation instruction was:

```powershell
.\scripts\stage06_3090_calibration.ps1 -ArtifactRoot '<actual-local-production-artifact-root>'
```

The script uses the checked-out commit as calibration code provenance, validates the committed artifact inventory and fixed nine-job batch proof, checks the pinned teacher/config/initialization hashes, requires the approved 3090 UUID as CUDA device 0, and writes raw 16-batch norm/ratio/factor evidence under the external `calibration-result` directory. It performs no optimizer updates and does not start production training. If the artifact tree is placed elsewhere, pass that absolute path to `-ArtifactRoot`.

| Input | Immutable identity |
|---|---|
| Corrected researcher amendment | `4660503cebe710f67f508e96ebd9d6002fcd19835951e773c4fb8062e1f4a097` |
| D01-D18 approval | `901ef13454dd07a40d90dc89d5db28b31f104897daa6907f54d2e4cc32ca2bae` |
| Reviewed nine-job batch candidate | `97fa155181fc9f251c3d7c09e07d8d1ed17794d65a21a68027315e290e7466b3` |
| Real external artifact inventory | `e4f515d26b0dbe1100ba97dc80c67eb224f7e7dc170c848a98123c0e19cc748f` |
| OpenWebText revision | `79d93d786212f7344586290adb811d4ae6a1762c` |
| Source JSONL SHA-256 | `d36784aaf0521f6e96d40d603e0361744397b23df90dc505e29b0b9cf18360eb` |
| Packed corpus payload | `76dcdc819b97ecff4d3c458bdf4dbd7c1a5dc2a525541e29099e6a6ac760d700` |
| Evaluation/calibration panels payload | `03fa2adf3931f7594fae23d70e402c697ad2c771e53bdb68977b614336cc5013` |
| Calibration block export payload | `306827da7f6987a02eadd716d43668fd4dceac1007cd7cb9b8bcdf4dd08b8413` |
| Seed-0 training order payload | `208b3bec7c15c1f8226972f8148d94b655b266b9faf7d3f0bc143eaa477e75f0` |
| Seed-0 student tensor content | `1a27890c588f317a4ae5523430f8af53c591bca29c9a78d13bf182c9c145fc09` |
| Seed-1729 calibration tensor content | `5fc6604f5eb8d2ff63958ec7a894736a727d56910db4fb68b865f203dbe93879` |
| GPT-2 tokenizer file set | `68bfbc36e7d352017e23168f34c98b3b92a18bdcb658c163dea499d8d690c580` |
| GPT-2-large teacher safetensors | `5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce` |

## 2026-09-28 RTX 3090 transfer and calibration result

The transfer archive `E:\KD-SINK-stage06-production-transfer.zip` (7,778,034,328 bytes) matched the 4080 receipt SHA-256 `220e428b54ba9c1e175826d3bcc14ac04ed0e5e2dcf9a59990d546ce32299d11` and remains intact. It restored the 27-file tree at `E:\KD-SINK-stage06-production`. The repository's full inventory validator produced external evidence at `E:\KD-SINK-stage06-3090\evidence\artifact-inventory-reverified.json`; every scientific field matches the committed inventory, with only the expected F:-to-E: absolute-root change.

From clean pushed commit `a29bcebc6253a5300452594bbaabe4b8e082a463`, the approved script ran for 198.9692297 seconds and wrote `E:\KD-SINK-stage06-production\calibration-result\s1-calibration-ddce66f76c43be3406f672cd2d0b4fd79fb35257ba28355caee72520e4932165.json` (sealed SHA-256 `ddce66f76c43be3406f672cd2d0b4fd79fb35257ba28355caee72520e4932165`). All 16 JSD/MSE/REL norms are positive finite; all 16 ratios per term are finite and recompute exactly. The registered medians are `s_MSE=68.00580071126464` and `s_REL=0.120179255876581`; C2/C5/C6 remain scale 1. Full arrays are preserved in the external result and `reports/stage06.json`. No optimizer update or S1 production training occurred.

Four measured component locks are sealed in `protocols/`. Stage 06 is still **blocked**, not production-ready: the approved records do not specify a positive `fingerprint_denominator_floor`, which the production evaluator requires. Until that value is explicitly approved, `protocol.lock.json`, the production root and the nine final configs remain absent. The date-only approval record is represented at date precision, without inventing a UTC approval time. No profiles were rerun; scientific coverage remains none.

The approved common schedule is microbatch 4 × accumulation 16 = 64 sequences of 128 tokens. C2 raw JSD, C3 raw MSE and C4 raw REL must be measured on the same seed-1729 CPU-FP32 initialization over the same 16 effective batches, with dropout disabled for calibration only and FP32 global norm reduction. The final locks and protocol root are sealed only after that measured result and the exact RTX 3090 environment are verified. Scientific coverage remains none.

## 2026-09-28 final sealing closure

The preceding blocked-state text is historical. The researcher explicitly approved
`fingerprint_denominator_floor=1e-8` on 2026-09-28, before any S1 production
training. It is only a numerical denominator guard for the reported S_I/S
ratio, not an effect-size threshold. When `abs(baseline_sink)<1e-8`, the ratio
is unavailable; baseline sink, probed sink and absolute sink difference remain.
The value is frozen independent of future results, matching the Stage 08
engineering guard without importing its fixture thresholds.

The sealer reverified the unchanged artifact/environment/hardware/calibration
component digests `2e721e9726e666cb8dc08a9b7d606a6d6ff99b434011b40fe2cf41f8eb315ccd`,
`0266e734fa97357ed0609e722c8d492284b34a06f726ce37b7301e703a370025`,
`c284abebf7b9915053c282d57555a1a28cf9ad8250f3c402a3f313207eaae138`,
and `fa031af3de63832e054b89539d3867c2904c9b592b1e7a8b3e4ca3e3676c532f`.
The final `protocol.lock.json` production root is
`2a11da9bb71957a4d6b3a2f93a34bd67491dd9d21d70577a7a10858930a8943e`.
All nine root-bound configs validate against that transitive lock set. Calibration
source commit `a29bcebc6253a5300452594bbaabe4b8e082a463` remains unchanged.
Stage 06 is production-ready; scientific coverage remains none and no 10k job
has started. The nine config SHA-256 values and final acceptance counts are
recorded in `reports/stage06.json`.
