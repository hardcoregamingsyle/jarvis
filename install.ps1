<#
.SYNOPSIS
    Installs JARVIS on Windows.

.DESCRIPTION
    Creates a virtual environment, installs the dependency profile you choose,
    downloads the British voice model, and verifies the result.

.PARAMETER Profile
    lean  - conversation, voice, memory, machine control (default; ~1 GB)
    full  - adds AirLLM + torch + transformers for a local Qwen3 (many GB)
    min   - the package only; everything optional is skipped

.PARAMETER NoVoiceDownload
    Skip fetching the Piper voice model.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Profile full
#>
[CmdletBinding()]
param(
    [ValidateSet('min', 'lean', 'full')]
    [string]$Profile = 'lean',
    [switch]$NoVoiceDownload,
    [string]$VenvPath = '.venv'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

function Write-Step($message) { Write-Host "`n==> $message" -ForegroundColor Cyan }
function Write-Ok($message)   { Write-Host "    $message" -ForegroundColor Green }
function Write-Warn($message) { Write-Host "    $message" -ForegroundColor Yellow }

Write-Host @"
    ___  _   _____   _   _ ___ ___
   |_ _|/_\ | _ \ \ / /_| / __/ __|
    | |/ _ \|   /\ V / | \__ \__ \
   |___/_/ \_\_|_\ \_/  |_|___/___/   installer for Windows
"@ -ForegroundColor DarkCyan

# --- Python ---------------------------------------------------------------
Write-Step 'Checking Python'
$python = $null
foreach ($candidate in @('py -3', 'python', 'python3')) {
    try {
        $parts = $candidate.Split(' ')
        $version = & $parts[0] $parts[1..($parts.Length - 1)] --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $version -match '(\d+)\.(\d+)') {
            if ([int]$Matches[1] -ge 3 -and [int]$Matches[2] -ge 9) {
                $python = $candidate
                Write-Ok "$version  ($candidate)"
                break
            }
        }
    } catch { }
}
if (-not $python) {
    Write-Error "Python 3.9+ is required. Install it from https://python.org and re-run."
    exit 1
}

# --- Virtual environment --------------------------------------------------
Write-Step "Creating the virtual environment ($VenvPath)"
if (-not (Test-Path $VenvPath)) {
    $parts = $python.Split(' ')
    & $parts[0] $parts[1..($parts.Length - 1)] -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { Write-Error 'Could not create the virtual environment.'; exit 1 }
    Write-Ok 'created'
} else {
    Write-Ok 'already present, reusing it'
}

$venvPython = Join-Path $root "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { Write-Error "Interpreter missing at $venvPython"; exit 1 }

Write-Step 'Upgrading pip'
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet
Write-Ok 'done'

# --- Dependencies ---------------------------------------------------------
Write-Step "Installing the '$Profile' profile"
switch ($Profile) {
    'min'  { & $venvPython -m pip install -e . }
    'lean' { & $venvPython -m pip install -e .; & $venvPython -m pip install -r requirements.txt }
    'full' {
        & $venvPython -m pip install -e .
        Write-Warn 'This downloads torch and friends - several gigabytes.'
        & $venvPython -m pip install -r requirements-full.txt
    }
}
if ($LASTEXITCODE -ne 0) {
    Write-Warn 'Some packages failed to install. JARVIS will still run in degraded mode.'
    Write-Warn "Run '$VenvPath\Scripts\jarvis doctor' to see exactly what is missing."
}

# --- Voice model ----------------------------------------------------------
if (-not $NoVoiceDownload -and $Profile -ne 'min') {
    Write-Step 'Fetching the British voice model'
    & $venvPython -m jarvis setup
} else {
    Write-Step 'Skipping the voice download'
    & $venvPython -m jarvis setup --no-download
}

# --- Launcher -------------------------------------------------------------
Write-Step 'Writing jarvis.bat'
@"
@echo off
rem Launch JARVIS using this project's virtual environment.
"%~dp0$VenvPath\Scripts\python.exe" -m jarvis %*
"@ | Set-Content -Path (Join-Path $root 'jarvis.bat') -Encoding ascii
Write-Ok 'jarvis.bat created'

# --- Done -----------------------------------------------------------------
Write-Host "`n==> Installed." -ForegroundColor Green
Write-Host @"

  Try it:
      .\jarvis.bat doctor        check what is installed
      .\jarvis.bat say           audition the British voice
      .\jarvis.bat chat          talk to it by keyboard
      .\jarvis.bat voice         hands-free

  A local language model is required for conversation. The quickest route:
      winget install Ollama.Ollama
      ollama pull qwen3:4b-instruct-2507-q4_K_M
  then set  llm.backend: ollama  in config.yaml.

"@ -ForegroundColor Gray
