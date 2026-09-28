param(
    [Parameter(Mandatory = $true)]
    [string] $ArtifactRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$artifactPath = (Resolve-Path -LiteralPath $ArtifactRoot).Path
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Locked Python environment is missing; run uv sync --locked on the approved RTX 3090.'
}

$candidateReport = Join-Path $repoRoot 'reports\stage06_reviewed_batch_candidate.json'
$inventory = Join-Path $repoRoot 'reports\stage06_production_artifact_inventory.json'
$partial = Join-Path $repoRoot 'protocols\s1_artifact_partial_v1.json'
$panels = Join-Path $artifactPath 'panels\owt-panels-03fa2adf3931f7594fae23d70e402c697ad2c771e53bdb68977b614336cc5013.json'
$calibrationPanel = Join-Path $artifactPath 'calibration-panel\calibration16x64-blocks-306827da7f6987a02eadd716d43668fd4dceac1007cd7cb9b8bcdf4dd08b8413.json'
$studentConfig = Join-Path $artifactPath 'student-config\config.json'
$initialization = Join-Path $artifactPath 'initialization\seed1729\init-seed1729-5fc6604f5eb8d2ff63958ec7a894736a727d56910db4fb68b865f203dbe93879.json'
$teacherDir = Join-Path $artifactPath 'teacher'
$outDir = Join-Path $artifactPath 'calibration-result'

Push-Location $repoRoot
try {
    $sourceCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve immutable source commit.' }
    & $pythonPath -m sinklab.stage06_calibration_entry `
        --candidate-report $candidateReport `
        --panel-export $calibrationPanel `
        --panels $panels `
        --artifact-inventory $inventory `
        --artifact-partial $partial `
        --teacher-dir $teacherDir `
        --student-config $studentConfig `
        --initialization $initialization `
        --out-dir $outDir `
        --source-commit $sourceCommit
    if ($LASTEXITCODE -ne 0) { throw "RTX 3090 calibration exited with status $LASTEXITCODE." }
} finally {
    Pop-Location
}
