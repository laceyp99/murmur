[CmdletBinding()]
param(
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "The Murmur release build must run on Windows."
}

$repoRoot = $PSScriptRoot
$pythonPath = Join-Path $repoRoot "venv\Scripts\python.exe"
$generatedPath = Join-Path $repoRoot "build\windows"
$executablePath = Join-Path $repoRoot "dist\Murmur\murmur.exe"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Create the repository venv before building: py -3.12 -m venv venv"
}

Push-Location $repoRoot
try {
    if (-not $SkipInstall) {
        & $pythonPath -m pip install -e ".[packaging]"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Murmur packaging dependencies."
        }
    }

    & $pythonPath tools\prepare_windows_build.py `
        --repo-root $repoRoot `
        --output-dir $generatedPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare Windows build resources."
    }

    & $pythonPath -m PyInstaller --noconfirm --clean packaging\murmur.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed to build Murmur."
    }

    if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
        throw "Expected executable was not created: $executablePath"
    }

    $versionInfo = (Get-Item -LiteralPath $executablePath).VersionInfo
    $expectedMetadata = @{
        FileDescription = "Murmur - Local Speech-to-Text Hotkey App"
        OriginalFilename = "murmur.exe"
        ProductName = "Murmur"
    }
    foreach ($field in $expectedMetadata.Keys) {
        if ($versionInfo.$field -ne $expectedMetadata[$field]) {
            throw "Executable metadata check failed for $field."
        }
    }

    $selfCheckLog = Join-Path $generatedPath "self-check-error.txt"
    Remove-Item -LiteralPath $selfCheckLog -ErrorAction SilentlyContinue
    $previousSelfCheckLog = $env:MURMUR_PACKAGING_SELF_CHECK_LOG
    $env:MURMUR_PACKAGING_SELF_CHECK_LOG = $selfCheckLog
    try {
        $selfCheck = Start-Process `
            -FilePath $executablePath `
            -ArgumentList "--packaging-self-check" `
            -WindowStyle Hidden `
            -PassThru
    }
    finally {
        $env:MURMUR_PACKAGING_SELF_CHECK_LOG = $previousSelfCheckLog
    }

    if (-not $selfCheck.WaitForExit(120000)) {
        Stop-Process -Id $selfCheck.Id -ErrorAction SilentlyContinue
        throw "Packaged dependency self-check timed out after 120 seconds."
    }
    if ($selfCheck.ExitCode -ne 0) {
        $failureDetail = if (Test-Path -LiteralPath $selfCheckLog -PathType Leaf) {
            Get-Content -LiteralPath $selfCheckLog -Raw
        }
        else {
            "No Python traceback was captured."
        }
        throw "Packaged dependency self-check failed with exit code $($selfCheck.ExitCode).`n$failureDetail"
    }
    Remove-Item -LiteralPath $selfCheckLog -ErrorAction SilentlyContinue

    Write-Host "Built and validated $executablePath"
    Write-Host "Product version: $($versionInfo.ProductVersion)"
}
finally {
    Pop-Location
}
