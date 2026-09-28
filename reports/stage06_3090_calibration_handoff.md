# Stage 06 RTX 3090 calibration handoff

The current host has only an RTX 4080 SUPER. Stage 06 remains blocked; this handoff does not report measured C3/C4 factors or a final production lock. On the approved RTX 3090 (`GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e`), pull the exact pushed `origin/main` source milestone, copy the external `F:\KD-SINK-stage06-production` tree to one local artifact root, and install the locked environment with `uv sync --locked`. Keep the checkout clean. Then invoke:

```powershell
.\scripts\stage06_3090_calibration.ps1 -ArtifactRoot 'F:\KD-SINK-stage06-production'
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

The approved common schedule is microbatch 4 × accumulation 16 = 64 sequences of 128 tokens. C2 raw JSD, C3 raw MSE and C4 raw REL must be measured on the same seed-1729 CPU-FP32 initialization over the same 16 effective batches, with dropout disabled for calibration only and FP32 global norm reduction. The final locks and protocol root are sealed only after that measured result and the exact RTX 3090 environment are verified. Scientific coverage remains none.
