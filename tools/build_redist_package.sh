#!/usr/bin/env bash
# build_redist_package.sh
#
# Creates a redistribution staging directory (redist_package/, NOT zipped)
# based on the list of top-level packages actually used, produced by
# packaging/trace_modules.py.
# Zipping is deliberately left out: this repository only ships
# hoops_ai_bridge/test_client, not the partner's actual client application,
# so a ZIP should only be produced once, together with the real client app,
# when the product itself is packaged. Compress the staging directory
# yourself (e.g. zip -r) at that time.
#
# Usage:
#   export HOOPS_AI_HOME=/path/to/HOOPS_AI/V1.1
#   ./tools/build_redist_package.sh \
#       --modules-file /tmp/used_top_level_modules.txt \
#       --ckpt path/to/ts3d_162k_mfr.ckpt \
#       --ckpt path/to/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt \
#       --sample path/to/sample_a.step \
#       --out-dir /path/to/staging_and_output_dir
#
# Prerequisites:
#   - bin/linux64/test_client and bin/linux64/libhoops_ai_bridge.so must already
#     be built (the binaries must have been built after placing hoops_license.h)
#   - $HOOPS_AI_HOME must point to a standard HOOPS AI installation. The
#     site-packages path is fixed at
#     $HOOPS_AI_HOME/.venv/lib/python3.12/site-packages, so it is built automatically.
#
# Note (disk space): The full target package set can be several GB in size
# (mainly due to hoops_exchange/hoops_converter/torch, etc.); check free
# space in --out-dir in advance (df -h).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULES_FILE=""
OUT_DIR=""
CKPT_FILES=()
SAMPLE_FILES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --modules-file) MODULES_FILE="$2"; shift 2 ;;
        --ckpt)         CKPT_FILES+=("$2"); shift 2 ;;
        --sample)       SAMPLE_FILES+=("$2"); shift 2 ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODULES_FILE" ] || [ -z "$OUT_DIR" ]; then
    echo "Usage: $0 --modules-file <trace_modules.py output> --out-dir <output directory> [--ckpt <ckpt file>]... [--sample <sample CAD file>]..." >&2
    exit 1
fi
# Build the venv site-packages path for a standard HOOPS AI installation from HOOPS_AI_HOME.
if [ -z "${HOOPS_AI_HOME:-}" ]; then
    echo "Please set the HOOPS_AI_HOME environment variable" >&2
    exit 1
fi
SOURCE_SITE_PACKAGES="$HOOPS_AI_HOME/.venv/lib/python3.12/site-packages"
if [ ! -f "$REPO_ROOT/bin/linux64/test_client" ]; then
    echo "Please build the project in $REPO_ROOT first (with hoops_license.h already in place)" >&2
    exit 1
fi

STAGE="$OUT_DIR/redist_package"
# Place python_packages using the same <home>/.venv/lib/python3.12/site-packages
# layout as a standard installation. test_client auto-detects this bundled
# .venv relative to its own location (<exeDir>/../.venv/...), so the package
# can be run directly as bin/test_client without setting HOOPS_AI_HOME or any
# launcher wrapper.
PYTHON_PACKAGES_DIR="$STAGE/.venv/lib/python3.12/site-packages"
mkdir -p "$STAGE/bin" "$PYTHON_PACKAGES_DIR" "$STAGE/models" "$STAGE/samples"

echo "[1/4] Copy native binaries"
cp "$REPO_ROOT/bin/linux64/test_client" "$STAGE/bin/"
cp "$REPO_ROOT/bin/linux64/libhoops_ai_bridge.so" "$STAGE/bin/" 2>/dev/null || \
cp "$REPO_ROOT/bin/win64/hoops_ai_bridge.dll" "$STAGE/bin/" 2>/dev/null || true

echo "[2/4] Copy checkpoints and sample CAD files"
for f in "${CKPT_FILES[@]:-}"; do [ -n "$f" ] && cp "$f" "$STAGE/models/"; done
for f in "${SAMPLE_FILES[@]:-}"; do [ -n "$f" ] && cp "$f" "$STAGE/samples/"; done

echo "[3/4] Copy only the required items from site-packages based on the trace results"

# Mapping for cases where the import name and distribution (wheel) name differ,
# causing the .libs directory name to differ as well.
# (Example: the import name is PIL, but the bundled native libraries are in
# pillow.libs.)
# If a new package causes ImportError: xxx.so: cannot open shared object file
# or similar, add an import-name -> .libs-directory-name mapping here.
declare -A LIBS_DIR_ALIAS=(
    [PIL]="pillow.libs"
    [sklearn]="scikit_learn.libs"
    [cv2]="opencv_python.libs"
    [faiss]="faiss_cpu.libs"
)

# List of Tech Soft 3D packages (redistributability must be confirmed
# individually under the TPA/license agreement).
# Other traced packages are treated as third-party dependencies such as OSS.
# If you start using additional Tech Soft 3D packages, add them here as well.
declare -A TS3D_CORE_PACKAGES=(
    [hoops_ai]=1
    [hoops_exchange]=1
    [hoops_converter]=1
    [hoops_web_viewer]=1
)

MANIFEST="$STAGE/THIRD_PARTY_MANIFEST.txt"
{
    echo "# Bundled Python package origin / redistributability manifest"
    echo "# Generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "#"
    echo "# [TechSoft3D] Packages provided by Tech Soft 3D. Whether they may be"
    echo "#   redistributed to end users depends on the specific terms of the"
    echo "#   TPA (partner agreement). Always confirm with your sales/licensing"
    echo "#   contact before distribution. Validation in this repository is only"
    echo "#   a technical runtime check and does not confirm license entitlement."
    echo "# [ThirdParty] Other third-party packages (mainly OSS). Many are usually"
    echo "#   low-risk for redistribution, but check individually whether each"
    echo "#   package requires LICENSE notices. Also refer to the dependency"
    echo "#   license list below:"
    echo "#   https://docs.techsoft3d.com/hoops/ai/resources/Acknowledgments.html"
    echo "#"
    echo "# Category,PackageName"
} > "$MANIFEST"

cd "$SOURCE_SITE_PACKAGES"
while read -r pkg; do
    [ -z "$pkg" ] && continue
    if [ -n "${TS3D_CORE_PACKAGES[$pkg]:-}" ]; then
        echo "TechSoft3D,$pkg" >> "$MANIFEST"
    else
        echo "ThirdParty,$pkg" >> "$MANIFEST"
    fi
    if [ -d "$pkg" ]; then
        cp -r "$pkg" "$PYTHON_PACKAGES_DIR/"
    else
        # If it is not a directory, look for a single-file module (*.so/*.py/*.pyi)
        for m in "$pkg".*; do
            [ -e "$m" ] && [[ "$m" != *dist-info ]] && cp "$m" "$PYTHON_PACKAGES_DIR/"
        done
    fi
    # Copy the corresponding dist-info directories as well (some libraries
    # require them for version lookup via importlib.metadata)
    for di in "${pkg}"-*.dist-info "${pkg}"*.dist-info; do
        [ -d "$di" ] && cp -r "$di" "$PYTHON_PACKAGES_DIR/" 2>/dev/null || true
    done
    # .libs support directories with the same name as the import, such as
    # numpy.libs / scipy.libs
    if [ -d "${pkg}.libs" ]; then
        cp -r "${pkg}.libs" "$PYTHON_PACKAGES_DIR/"
    fi
    # Cases where the import name and .libs directory name differ
    # (mapping table above)
    if [ -n "${LIBS_DIR_ALIAS[$pkg]:-}" ] && [ -d "${LIBS_DIR_ALIAS[$pkg]}" ]; then
        cp -r "${LIBS_DIR_ALIAS[$pkg]}" "$PYTHON_PACKAGES_DIR/"
    fi
done < "$MODULES_FILE"
cd - > /dev/null
echo "  -> Generated manifest: $MANIFEST"

echo "[4/4] Prepare README"
cp "$REPO_ROOT/tools/redist_README.md.example" "$STAGE/README.md"
# No launcher wrapper is generated: test_client auto-detects the bundled
# .venv relative to its own path, so run it directly as bin/test_client
# (on a headless Linux server, start Xvfb and export DISPLAY first; see README).

echo "[Done] Staging directory ready (not zipped)"
du -sh "$STAGE"
echo "  -> $STAGE"
echo "(This is only the staging directory for hoops_ai_bridge/test_client."
echo " Zip it together with your actual client application once that is"
echo " ready, rather than zipping this alone.)"
echo "(See $STAGE/THIRD_PARTY_MANIFEST.txt for bundled package origin and redistribution category)"
