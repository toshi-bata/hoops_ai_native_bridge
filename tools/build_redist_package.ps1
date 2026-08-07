<#
.SYNOPSIS
    Windows (PowerShell) version of build_redist_package.sh.

.DESCRIPTION
    Creates a redistribution staging directory (redist_package\, NOT zipped)
    based on the list of top-level packages actually used, produced by
    tools/trace_modules.py.
    Zipping is deliberately left out: this repository only ships
    hoops_ai_bridge/test_client, not the partner's actual client
    application, so a ZIP should only be produced once, together with the
    real client app, when the product itself is packaged. Compress the
    staging directory yourself (e.g. Compress-Archive) at that time.

.PARAMETER ModulesFile
    Module list file output by trace_modules.py --out.

.PARAMETER OutDir
    Output directory for the staging directory and ZIP.

.PARAMETER Ckpt
    Checkpoint (.ckpt) files to include. To specify multiple files, pass a
    comma-separated array (example: -Ckpt "a.ckpt","b.ckpt"). The same
    parameter cannot be specified multiple times.

.PARAMETER Sample
    Sample CAD files to include. To specify multiple files, pass a
    comma-separated array.

.EXAMPLE
    $env:HOOPS_AI_HOME = "<path to HOOPS_AI_HOME>"
    .\tools\build_redist_package.ps1 `
        -ModulesFile C:\temp\used_top_level_modules.txt `
        -Ckpt "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_162k_mfr.ckpt", "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" `
        -Sample C:\temp\sample_a.step `
        -OutDir C:\temp\redist_out

.NOTES
    Prerequisites:
      - bin\win64\test_client.exe and bin\win64\hoops_ai_bridge.dll
        must already be built
        (the binaries must have been built after placing hoops_license.h)
      - $env:HOOPS_AI_HOME must point to a standard HOOPS AI installation.
        The site-packages path is fixed at
        $env:HOOPS_AI_HOME\.venv\Lib\site-packages, so it is built automatically.

    Note (disk space): The full target package set can be several GB in size
    (mainly due to hoops_exchange/hoops_converter/torch, etc.); check free
    space in $OutDir in advance.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ModulesFile,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string[]]$Ckpt = @(),
    [string[]]$Sample = @()
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $ModulesFile)) {
    throw "modules file not found: $ModulesFile"
}
# Build the venv site-packages path for a standard HOOPS AI installation from HOOPS_AI_HOME.
if (-not $env:HOOPS_AI_HOME) {
    throw "Please set the HOOPS_AI_HOME environment variable"
}
$SitePackages = Join-Path $env:HOOPS_AI_HOME ".venv\Lib\site-packages"
$TestClientExe = Join-Path $RepoRoot "bin\win64\test_client.exe"
if (-not (Test-Path $TestClientExe)) {
    throw "Please build the project in $RepoRoot first (with hoops_license.h already in place)"
}

$Stage = Join-Path $OutDir "redist_package"
# Place python_packages using the same <home>\.venv\Lib\site-packages layout as
# a standard installation. test_client auto-detects this bundled .venv relative
# to its own location (<exeDir>\..\.venv\...), so the package can be run directly
# as bin\test_client.exe without setting HOOPS_AI_HOME or any launcher wrapper.
$PythonPackagesDir = "$Stage\.venv\Lib\site-packages"
New-Item -ItemType Directory -Force -Path "$Stage\bin", $PythonPackagesDir, "$Stage\models", "$Stage\samples" | Out-Null

Write-Host "[1/4] Copy native binaries"
Copy-Item $TestClientExe "$Stage\bin\" -Force
$BridgeDll = Join-Path $RepoRoot "bin\win64\hoops_ai_bridge.dll"
if (Test-Path $BridgeDll) { Copy-Item $BridgeDll "$Stage\bin\" -Force }

Write-Host "[2/4] Copy checkpoints and sample CAD files"
foreach ($f in $Ckpt) { if ($f) { Copy-Item $f "$Stage\models\" -Force } }
foreach ($f in $Sample) { if ($f) { Copy-Item $f "$Stage\samples\" -Force } }

Write-Host "[3/4] Copy only the required items from site-packages based on the trace results"

# Mapping for cases where the import name and distribution (wheel) name differ,
# causing the .libs directory name to differ as well.
# (Example: the import name is PIL, but the bundled native libraries are in
# pillow.libs.)
# If a new package causes ImportError: DLL load failed or similar, add an
# import-name -> .libs-directory-name mapping here.
$LibsDirAlias = @{
    "PIL"     = "pillow.libs"
    "sklearn" = "scikit_learn.libs"
    "cv2"     = "opencv_python.libs"
    "faiss"   = "faiss_cpu.libs"
}

# List of Tech Soft 3D packages (redistributability must be confirmed
# individually under the TPA/license agreement).
# Other traced packages are treated as third-party dependencies such as OSS.
# If you start using additional Tech Soft 3D packages, add them here as well.
$Ts3dCorePackages = @{
    "hoops_ai"         = $true
    "hoops_exchange"   = $true
    "hoops_converter"  = $true
    "hoops_web_viewer" = $true
}

$Manifest = Join-Path $Stage "THIRD_PARTY_MANIFEST.txt"
@(
    "# Bundled Python package origin / redistributability manifest"
    "# Generated at: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    "#"
    "# [TechSoft3D] Packages provided by Tech Soft 3D. Whether they may be"
    "#   redistributed to end users depends on the specific terms of the"
    "#   TPA (partner agreement). Always confirm with your sales/licensing"
    "#   contact before distribution. Validation in this repository is only"
    "#   a technical runtime check and does not confirm license entitlement."
    "# [ThirdParty] Other third-party packages (mainly OSS). Many are usually"
    "#   low-risk for redistribution, but check individually whether each"
    "#   package requires LICENSE notices. Also refer to the dependency"
    "#   license list below:"
    "#   https://docs.techsoft3d.com/hoops/ai/resources/Acknowledgments.html"
    "#"
    "# Category,PackageName"
) | Set-Content -Path $Manifest -Encoding UTF8

Push-Location $SitePackages
try {
    foreach ($pkg in (Get-Content $ModulesFile | Where-Object { $_.Trim() -ne "" })) {
        if ($Ts3dCorePackages.ContainsKey($pkg)) {
            Add-Content -Path $Manifest -Value "TechSoft3D,$pkg"
        } else {
            Add-Content -Path $Manifest -Value "ThirdParty,$pkg"
        }

        if (Test-Path $pkg -PathType Container) {
            Copy-Item $pkg $PythonPackagesDir -Recurse -Force
        } else {
            # If it is not a directory, look for a single-file module (*.pyd/*.py/*.pyi)
            Get-ChildItem -Path "." -Filter "$pkg.*" -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notlike "*dist-info*" } |
                ForEach-Object { Copy-Item $_.FullName $PythonPackagesDir -Force }
        }

        # Copy the corresponding dist-info directories as well (some libraries
        # require them for version lookup via importlib.metadata)
        Get-ChildItem -Path "." -Directory -Filter "$pkg-*.dist-info" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName $PythonPackagesDir -Recurse -Force }
        Get-ChildItem -Path "." -Directory -Filter "$pkg*.dist-info" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName $PythonPackagesDir -Recurse -Force }

        # .libs support directories with the same name as the import, such as
        # numpy.libs / scipy.libs
        if (Test-Path "$pkg.libs" -PathType Container) {
            Copy-Item "$pkg.libs" $PythonPackagesDir -Recurse -Force
        }
        # Cases where the import name and .libs directory name differ
        # (mapping table above)
        if ($LibsDirAlias.ContainsKey($pkg) -and (Test-Path $LibsDirAlias[$pkg] -PathType Container)) {
            Copy-Item $LibsDirAlias[$pkg] $PythonPackagesDir -Recurse -Force
        }
    }
} finally {
    Pop-Location
}
Write-Host "  -> Generated manifest: $Manifest"

Write-Host "[4/4] Prepare README"
Copy-Item (Join-Path $RepoRoot "tools\redist_README.md.example") (Join-Path $Stage "README.md") -Force
# No launcher wrapper is generated: test_client auto-detects the bundled .venv
# relative to its own path, so run it directly as bin\test_client.exe.

Write-Host "[Done] Staging directory ready (not zipped)"
$stageSizeGB = (Get-ChildItem $Stage -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("  Staging size: {0:N2} GB" -f $stageSizeGB)
Write-Host ("  -> {0}" -f $Stage)
Write-Host "(This is only the staging directory for hoops_ai_bridge/test_client."
Write-Host " Zip it together with your actual client application once that is"
Write-Host " ready, rather than zipping this alone.)"
Write-Host "(See $Stage\THIRD_PARTY_MANIFEST.txt for bundled package origin and redistribution category)"
