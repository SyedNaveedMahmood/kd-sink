param(
    [string]$Python = "",
    [string]$LogDirectory = "results/_nnsight_rest_smoke/logs"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
# transformers writes its "Loading weights"/"Writing model shards" progress bars to
# stderr. Silencing them keeps the logs readable, and — with the ErrorActionPreference
# handling below — keeps a cosmetic bar from being mistaken for a failure.
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Logs = Join-Path $RepoRoot $LogDirectory
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

# Default to the repo's own venv rather than to whatever `python` resolves to. A bare
# `python` on a fresh shell finds the system interpreter, which has no nnsight, and the
# run dies inside E3 with a ModuleNotFoundError that reads like a code fault rather than a
# wrong interpreter. Pass -Python explicitly to override.
if (-not $Python) {
    $Venv = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $Venv) { $Python = $Venv } else { $Python = "python" }
}
Write-Host ("Interpreter: {0}" -f $Python)

$Tests = @(
    @{ Name = "E3"; Path = "tests/nnsight_e3_smoke.py" },
    @{ Name = "E4"; Path = "tests/nnsight_e4_smoke.py" },
    @{ Name = "E5"; Path = "tests/nnsight_e5_smoke.py" },
    # E6 (WP3) and E7 (WP7) were added later; pytest does not collect nnsight_*_smoke.py,
    # so this runner is the only thing that executes them.
    @{ Name = "E6"; Path = "tests/nnsight_e6_smoke.py" },
    @{ Name = "E7"; Path = "tests/nnsight_e7_smoke.py" },
    # E6-EVAL (WP10) runs the whole E6 pipeline offline: train -> evaluate -> aggregate
    # -> pilot gate. Slower than the others because it trains three conditions.
    @{ Name = "E6-EVAL"; Path = "tests/nnsight_e6_eval_smoke.py" },
    # E7-PIPE (WP9 + WP11) runs the whole E7 pipeline offline: manifests -> extraction ->
    # retrieval -> multilingual fingerprints -> patching (screen + window) -> aggregation.
    @{ Name = "E7-PIPE"; Path = "tests/nnsight_e7_pipeline_smoke.py" },
    # E6B (WP6) trains three F-conditions, merges the LoRA adapters and proves the
    # *_drift_from_base and task columns reach the aggregator's E6B tables.
    @{ Name = "E6B"; Path = "tests/nnsight_e6b_smoke.py" },
    # E6A-GPT2 runs the second distillation arm offline on a tiny GPT-2 12->6 pair, and
    # asserts the property the arm split exists for: its rows never reach the TinyStories
    # arm's aggregation.
    @{ Name = "E6A-GPT2"; Path = "tests/nnsight_e6a_gpt2_smoke.py" },
    # E6A-MEDIUM-SMALL preserves the registered 24/16 -> 12/12 geometry at scaled width,
    # trains the AMAD-style JSD arm and its equal-head legacy/aligned bridge, then runs the
    # evaluator, aggregator and arm-specific gate entirely offline.
    @{ Name = "E6A-MEDIUM-SMALL"; Path = "tests/nnsight_e6a_gpt2_medium_small_smoke.py" }
)

$OriginalLocation = Get-Location
try {
    Set-Location $RepoRoot
    foreach ($Test in $Tests) {
        $Log = Join-Path $Logs ("{0}_smoke.log" -f $Test.Name.ToLowerInvariant())
        Write-Host ("==== Running {0} offline NNsight smoke test ====" -f $Test.Name)
        # `2>&1` on a native command wraps every stderr line in a NativeCommandError, which
        # under `$ErrorActionPreference = "Stop"` terminates the run even when python
        # exited 0. The merge is wanted (the log should hold both streams), so the
        # preference is relaxed for the call only and the real verdict is taken from
        # $LASTEXITCODE, which is the process's own exit code and nothing else.
        $Previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python $Test.Path 2>&1 | Tee-Object -FilePath $Log
            $Code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $Previous
        }
        if ($Code -ne 0) {
            throw ("{0} smoke test failed with exit code {1}. Log: {2}" -f $Test.Name, $Code, $Log)
        }
    }
    Write-Host ("==== All E3/E4/E5/E6/E7/E6-EVAL/E7-PIPE/E6B/E6A-GPT2/" +
                "E6A-MEDIUM-SMALL offline NNsight smoke tests passed ====")
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Set-Location $OriginalLocation
}
exit 0
