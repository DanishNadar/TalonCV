param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$hfPath = Join-Path $projectRoot ".venv\Scripts\hf.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Create .venv and install requirements.txt before setting up local models."
}
if (-not (Test-Path -LiteralPath $hfPath)) {
    throw "The setup-only hf CLI is missing. Run: .venv\Scripts\python.exe -m pip install -r requirements-model-setup.txt"
}

function Test-LocalModel {
    param([string]$Name)
    & $pythonPath (Join-Path $projectRoot "scripts\verifyLocalModels.py") --model $Name --files-only --quiet
    return $LASTEXITCODE -eq 0
}

function Install-LocalModel {
    param(
        [string]$Name,
        [string]$Repository,
        [string]$Destination,
        [string]$Include = ""
    )
    if (-not $Force -and (Test-LocalModel $Name)) {
        Write-Host "$Name is already complete; skipping."
        return
    }
    $destinationPath = Join-Path $projectRoot $Destination
    New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
    $arguments = @("download", $Repository, "--local-dir", $destinationPath)
    if ($Include) {
        $arguments += @("--include", $Include)
    }
    if ($Force) {
        $arguments += "--force-download"
    }
    Write-Host "Downloading $Name to $destinationPath"
    & $hfPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Model setup failed for $Name."
    }
    if (-not (Test-LocalModel $Name)) {
        throw "$Name downloaded, but required local files are incomplete."
    }
}

Set-Location $projectRoot
Install-LocalModel "transcription" "Systran/faster-whisper-small.en" "models\faster-whisper-small.en"
Install-LocalModel "faceDetection" "AdamCodd/YOLOv11n-face-detection" "models\yolo11n-face" "*.pt"
Install-LocalModel "semanticAnalysis" "sentence-transformers/all-MiniLM-L6-v2" "models\all-MiniLM-L6-v2"
Install-LocalModel "localCoach" "Qwen/Qwen2.5-1.5B-Instruct" "models\qwen2.5-1.5b-instruct"

& $pythonPath (Join-Path $projectRoot "scripts\verifyLocalModels.py") --files-only
if ($LASTEXITCODE -ne 0) {
    throw "One or more local model directories failed verification."
}
Write-Host "Local model setup complete. Run scripts\verifyLocalModels.py without --files-only to test actual loading."
