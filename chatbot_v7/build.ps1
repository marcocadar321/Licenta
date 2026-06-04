

param(
    [switch]$OnlyBuild,
    [switch]$OnlyServer
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  ChatBot NLP v7 - Build & Start" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""


if (-not $OnlyServer) {
    Write-Host "[1/3] Verific g++..." -ForegroundColor Yellow
    if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
        Write-Host "  EROARE: g++ nu este instalat sau nu e in PATH." -ForegroundColor Red
        Write-Host "  Descarca MinGW-w64 de la https://winlibs.com" -ForegroundColor Red
        exit 1
    }
    Write-Host "  g++ gasit." -ForegroundColor Green

    Write-Host "[2/3] Compilez preprocess.dll..." -ForegroundColor Yellow
    if (-not (Test-Path "build")) {
        New-Item -ItemType Directory -Path "build" | Out-Null
    }

    g++ -O2 -shared -o build/preprocess.dll preprocess.cpp
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  EROARE: Compilarea a esuat." -ForegroundColor Red
        exit 1
    }
    Write-Host "  build/preprocess.dll compilat cu succes." -ForegroundColor Green
}

if ($OnlyBuild) {
    Write-Host ""
    Write-Host "Build complet. Ruleaza '.\build.ps1 -OnlyServer' pentru a porni serverul." -ForegroundColor Cyan
    exit 0
}


Write-Host "[3/3] Pornesc serverul..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Server: http://localhost:8000" -ForegroundColor Green
Write-Host "  Oprire: CTRL+C" -ForegroundColor Green
Write-Host ""

uvicorn api:app --reload --port 8000
