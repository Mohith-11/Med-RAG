# pull_models.ps1
# ================================================================================
# Downloads all 6 medical LLM generators via Ollama.
# Run this script ONCE before compare_generators.py.
#
# Usage (from rag_project root):
#   .\scripts\pull_models.ps1              # pull all 6 models
#   .\scripts\pull_models.ps1 meditron     # pull one specific model only
# ================================================================================

param(
    [string]$OnlyModel = ""
)

$ErrorActionPreference = "Stop"

# ── Model registry (must match generator/model_registry.py) ──────────────────
$Models = [ordered]@{
    "medgemma"    = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
    "meditron"    = "meditron"
    "medalpaca"   = "medalpaca"
    "biomistral"  = "biomistral"
    "llama3med42" = "llama3-med42"
    "pmcllama"    = "hf.co/bartowski/pmc-llama-13b-GGUF:Q4_K_M"
}

$Sizes = @{
    "medgemma"    = "~2.5 GB"
    "meditron"    = "~4.1 GB"
    "medalpaca"   = "~3.8 GB"
    "biomistral"  = "~4.5 GB"
    "llama3med42" = "~5.0 GB"
    "pmcllama"    = "~7.9 GB"
}

# ── Helper ────────────────────────────────────────────────────────────────────
function Write-Banner($msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Test-OllamaRunning {
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
Write-Banner "Oncology RAG — Medical LLM Model Downloader"

# Check ollama is on PATH
if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] 'ollama' not found on PATH. Install from https://ollama.com/download" -ForegroundColor Red
    exit 1
}

# Check Ollama daemon is running
if (-not (Test-OllamaRunning)) {
    Write-Host "[WARN]  Ollama daemon not detected on localhost:11434." -ForegroundColor Yellow
    Write-Host "        Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
    if (-not (Test-OllamaRunning)) {
        Write-Host "[ERROR] Could not start Ollama. Run 'ollama serve' manually first." -ForegroundColor Red
        exit 1
    }
}
Write-Host "[OK]    Ollama daemon is running." -ForegroundColor Green

# ── Build pull list ───────────────────────────────────────────────────────────
if ($OnlyModel -ne "") {
    if (-not $Models.Contains($OnlyModel)) {
        Write-Host "[ERROR] Unknown model key '$OnlyModel'." -ForegroundColor Red
        Write-Host "        Valid keys: $($Models.Keys -join ', ')" -ForegroundColor Yellow
        exit 1
    }
    $ToPull = [ordered]@{ $OnlyModel = $Models[$OnlyModel] }
} else {
    $ToPull = $Models
}

# ── Pull loop ─────────────────────────────────────────────────────────────────
$Total   = $ToPull.Count
$Current = 0
$Failed  = @()

foreach ($Key in $ToPull.Keys) {
    $Current++
    $Tag  = $ToPull[$Key]
    $Size = $Sizes[$Key]

    Write-Host ""
    Write-Host "[$Current/$Total] Pulling: $Key  ($Tag)  $Size" -ForegroundColor Yellow

    try {
        & ollama pull $Tag
        if ($LASTEXITCODE -ne 0) { throw "ollama pull exited with code $LASTEXITCODE" }
        Write-Host "[OK]    $Key pulled successfully." -ForegroundColor Green
    } catch {
        Write-Host "[FAIL]  $Key failed: $_" -ForegroundColor Red
        $Failed += $Key
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
if ($Failed.Count -eq 0) {
    Write-Host "  All models pulled successfully!" -ForegroundColor Green
} else {
    Write-Host "  Completed with errors. Failed models:" -ForegroundColor Yellow
    $Failed | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  Retry individual models with:" -ForegroundColor Yellow
    $Failed | ForEach-Object { Write-Host "    .\scripts\pull_models.ps1 $_" -ForegroundColor Gray }
}
Write-Host ""
Write-Host "  Run evaluation:" -ForegroundColor Cyan
Write-Host "    python compare_generators.py --models meditron" -ForegroundColor Gray
Write-Host "    python compare_generators.py --models medgemma meditron medalpaca" -ForegroundColor Gray
Write-Host ("=" * 70) -ForegroundColor Cyan
