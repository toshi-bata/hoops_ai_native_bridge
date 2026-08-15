# HOOPS AI native bridge

A reference implementation that lets a **native Windows/Linux application call HOOPS AI in-process** — no
web server, no HTTP, no separate Python process. A single shared library embeds a CPython interpreter,
imports `hoops_ai`, and exposes a plain **C ABI**, so the calling application never sees Python or PyTorch.

The library is delivered the way every other HOOPS SDK is — as an `include` / `lib` / `bin` package:
include the header, link against `lib` (Windows), and ship the DLL / shared object from `bin` alongside
your own binaries.

> **This is a starting point, not a product.** It deliberately does not cover every HOOPS AI feature, and
> the pretrained models used here (`ts3d_162k_mfr.ckpt`, `ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt`)
> are the ones bundled with the HOOPS AI tutorials. A shipping product needs its own added value — models
> retrained on your own CAD data, your own inference logic, UI integration — rather than the tutorial
> models as-is. Use this repository as a reference for building your own bridge, keeping the features your
> product needs and adding your own.

## What has been verified

- MFR (per-face feature classification), shape similarity, similar-part search (FAISS), and
  assembly-to-assembly search, all called from a native C++ client.
- Windows (x86_64) and Linux (Ubuntu 24.04) builds from the same source.
- **Redistribution**: the built client, the bridge, and the required subset of HOOPS AI's `site-packages`,
  packaged and run on a machine with **no HOOPS AI installation** and no reference to the development venv.
- CPU inference only; GPU is untested.

## How it works

### Architecture

```text
Client application  (C/C++, C#, Qt, MFC, WPF, console)
        |  C ABI  (plain C functions only)
hoops_ai_bridge     (.dll / .so — embeds one CPython interpreter)
        |  import
HOOPS AI            (Python / PyTorch)
```

All three layers run in the same process.

- One CPython interpreter is initialized inside the library (`Py_InitializeEx`), the venv's
  `site-packages` is added to `sys.path`, and `hoops_ai` is imported normally.
- `include/hoops_ai_bridge.h` is the **single source of truth** for the exported `HoopsAI_*` functions and
  their contracts. The set grows as the bridge gains features, so it is intentionally not duplicated here.
- **CAD access is shared.** MFR (`FlowInference`) and shape embeddings (`HOOPSEmbeddings`) both need a
  `HOOPSLoader`. The bridge creates one lazily and hands the same instance to both
  (`GetSharedCADLoader()` in `src/hoops_ai_bridge.cpp`) instead of constructing a loader per call — the
  `HOOPSEmbeddings` constructor accepts an optional `cad_loader`, which is what makes this possible.

### The C ABI is a command-level API, not a Python wrapper

This is a deliberate design principle. Each exported `HoopsAI_*` function is **one high-level operation the
client thinks in** — "add this CAD file to the index and render its thumbnail", "search similar
assemblies" — and the bridge internally orchestrates the several `hoops_ai` calls that operation needs
(loader creation, embedding and batching, FAISS upsert plus atomic save, thumbnail generation, matcher
construction, and so on).

That keeps the caller free of Python / PyTorch / FAISS concepts, turns what would be many chained SDK calls
into one call, and lets the bridge change its internals without breaking the client contract. When
productizing, keep shaping these commands around **your product's features** rather than re-exposing HOOPS
AI internals as-is.

### Two ways to bridge a feature

Not everything you may want to bridge lives inside the compiled `hoops_ai` package. Which pattern applies
is decided purely by whether the feature exists as an importable module:

- **Pattern A — import the compiled `hoops_ai` API** (the default). The bridge imports the relevant
  submodule (`hoops_ai.ml.embeddings`, `hoops_ai.cadaccess`, `hoops_ai.ml.CADSearch`, ...) and drives the
  resulting objects from C++. Used by MFR, shape embeddings, and the FAISS index operations.

- **Pattern B — embed tutorial Python source and `exec` it** (currently only assembly-to-assembly search).
  `AssemblyMatcher` — the reference implementation of assembly retrieval (part-level Hungarian matching,
  TF-IDF rare-part weighting, bag-of-parts composition) — ships as *tutorial source*
  (`HOOPS-AI-tutorials/embeddings_pipeline/assembly_matcher.py`), not as an importable module. There is
  nothing to import, so the source itself has to be carried: it is captured verbatim as a C string literal
  in `src/assembly_matcher_py.h` and `PyRun_String`-exec'd once, on first use, into a private namespace.
  A per-index matcher cache makes the one-off IDF / k-means build a first-call-only cost.

  > **Trade-off**: the embedded copy must be re-synced whenever the upstream tutorial changes, and nothing
  > catches drift at compile time. `src/assembly_matcher_py.h` is **auto-generated from the tutorial file**
  > — never hand-edit it; edit the tutorial and regenerate. If a future release promotes `AssemblyMatcher`
  > into the compiled `hoops_ai` package, this feature should move to Pattern A.

### Good to know

- `hoops_ai` is itself a **native extension compiled with Nuitka** (Windows:
  `hoops_ai.cp312-win_amd64.pyd`, Linux: `hoops_ai.cpython-312-x86_64-linux-gnu.so`; filenames vary by
  version). The source is closed, but `import hoops_ai` works normally.
- The inference engine is **PyTorch / PyTorch Geometric**, not ONNX Runtime. So the realistic approach is
  not "bring only the model into C++" but "embed Python along with it".
- [Supported platforms](https://docs.techsoft3d.com/hoops/ai/getting_started/supported_platforms.html):
  Windows x86_64 and GNU/Linux (x86_64, arm64-v8a; Ubuntu 22.04 LTS+) — which is why the bridge is
  cross-platform from the start.

## Repository layout

```text
hoops_ai_native_bridge/
├── CMakeLists.txt              # Builds hoops_ai_bridge only (Windows/Linux). Does not build the sample.
├── include/
│   ├── hoops_ai_bridge.h       # C ABI — the API reference for callers
│   ├── hoops_license.h.example # Template for embedding a license key at build time
│   └── hoops_license.h         # Your key (copy of the above; .gitignore'd)
├── src/
│   ├── hoops_ai_bridge.cpp     # CPython embedding + hoops_ai calls
│   └── assembly_matcher_py.h   # Auto-generated: AssemblyMatcher tutorial source as a C string
├── samples/
│   ├── CMakeLists.txt          # Standalone build for test_client (links the already-built bridge)
│   └── test_client.cpp         # Console client used to verify each feature
├── lib/win64/
│   └── hoops_ai_bridge.lib     # Windows import library (committed, so external projects can link)
├── bin/
│   ├── win64/                  # hoops_ai_bridge.dll (committed) + test_client.exe (never committed)
│   └── linux64/                # libhoops_ai_bridge.so (committed) + test_client (never committed)
└── tools/
    ├── trace_modules.py        # Measures which Python modules your features actually import
    ├── build_redist_package.ps1 / .sh   # Assembles the redistribution staging directory
    ├── redist_README.md.example # Template README to ship inside the package
    └── repro_plain_python.py   # Debug aid: run the same calls in plain Python for a full traceback
```

> **Security note**: only the shared libraries and the Windows import library are committed. Their public C
> ABI embeds no license key — the caller supplies it at runtime. `test_client` **must never be committed**:
> it compiles in an actual license key from `include/hoops_license.h`. `.gitignore` excludes it (along with
> the `.exp` / `.pdb` files); check `git status` before pushing.

## Prerequisites

HOOPS AI already installed by the official procedure — that is, a Python 3.12 venv created with
`pip install "hoops-ai[all]"`. Python 3.12 is required by that installation, so no separate Python setup is
needed. Below, `HOOPS_AI_HOME` means the installation directory: the parent of `.venv/` and
`packages/trained_ml_models/`, for example `C:\SDK\HOOPS_AI\V1.1` or `/opt/hoops_ai/V1.1`.

The target Linux distribution is **Ubuntu 24.04 LTS**, whose default Python is 3.12. Install the build
tools first:

```bash
sudo apt install -y cmake g++ python3.12-dev
```

- `cmake` is not part of a minimal install; without it `cmake -S . -B build` aborts before configuring.
- `g++` is missing from some minimal installs (`gcc` may exist while `g++` does not), which fails CMake
  with `No CMAKE_CXX_COMPILER could be found`.
- `python3.12-dev` provides the headers and the shared `libpython3.12.so` that
  `find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Development)` needs — the venv alone is not enough.
  On Ubuntu 24.04 this is a stock apt package built with `--enable-shared`, so the bridge (a `.so`) links
  against it cleanly.

> Not on 24.04? Where 3.12 is not the default Python, `python3.12-dev` may be unavailable via apt and a
> hand-built Python must be configured with `--enable-shared` (a static-only `libpython3.12.a` fails to link
> into a shared library with `relocation R_X86_64_TPOFF32 ... can not be used when making a shared object`).

> On a **headless** Linux machine, license validation can return `INVALID` when `DISPLAY` is unset; that
> setup is covered separately.

## Build

Two independent steps. Step 1 alone is enough if you only need the redistributable bridge library — for
example to link it from a Qt project.

### Step 1: the bridge

**Windows** (Visual Studio 2022; either the Developer Command Prompt or PowerShell works):

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

On Windows, `find_package(Python3 3.12 EXACT ...)` auto-detects a registered Python 3.12 (python.org
installers register themselves in the PEP 514 registry) including its `Development` component, so
**`-DPython3_ROOT_DIR` is normally unnecessary**. Pass it only if auto-detection fails or picks the wrong
interpreter — for instance with several Python 3.12 installations, or one whose development files are
missing:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DPython3_ROOT_DIR="<Python 3.12 install dir>"
```

Don't hardcode that path: the python.org installer's location varies by version and options (per-user,
all-users, or the newer "Python Install Manager"). Confirm it with

```powershell
py -3.12 -c "import sys; print(sys.base_prefix)"
```

**Linux** — activate the existing HOOPS AI venv first (don't create a new one). CMake finds `python3` on
PATH, so `-DPython3_ROOT_DIR` is generally unnecessary here:

```bash
export HOOPS_AI_HOME=/path/to/HOOPS_AI/V1.1
source "$HOOPS_AI_HOME/.venv/bin/activate"
python -c "import hoops_ai; print(hoops_ai.__version__)"   # sanity check

cmake -S . -B build
cmake --build build
```

**Verify.** The build writes directly to `bin/win64/` and `lib/win64/` (or `bin/linux64/`) —
`CMakeLists.txt` overrides the per-configuration output paths, so nothing lands under `build/Release/`.

**Windows**:

```powershell
dir bin\win64\hoops_ai_bridge.dll, lib\win64\hoops_ai_bridge.lib
```

**Linux**:

```bash
ls -l bin/linux64/libhoops_ai_bridge.so
```

### Step 2: the test client

`samples/` is a **separate, standalone CMake project**; the root build does not include it. Build it after
step 1.

First place your license key. This is required for the client, not for the bridge — and there is no
environment-variable fallback, so the key has to be in this header.

**Windows**:

```bat
copy include\hoops_license.h.example include\hoops_license.h
:: then edit include\hoops_license.h and set:  #define HOOPS_LICENSE "your actual key"
```

**Linux**:

```bash
cp include/hoops_license.h.example include/hoops_license.h
# then edit include/hoops_license.h and set:  #define HOOPS_LICENSE "your actual key"
```

Then build. **Windows**:

```bat
cmake -S samples -B samples/build -G "Visual Studio 17 2022" -A x64
cmake --build samples/build --config Release
```

**Linux**:

```bash
cmake -S samples -B samples/build
cmake --build samples/build
```

No `-DPython3_ROOT_DIR` here: the sample only links the already-built bridge, it does not embed Python
itself. The binary lands in `bin/win64/test_client.exe` or `bin/linux64/test_client`, next to the bridge
library.

## Running the test client

Set `HOOPS_AI_HOME` at runtime; the client derives `site-packages` from it
(`<HOOPS_AI_HOME>\.venv\Lib\site-packages` on Windows,
`<HOOPS_AI_HOME>/.venv/lib/python3.12/site-packages` on Linux). The license key is already compiled in.

`HOOPS_AI_PYTHON_HOME` is **optional on Windows**: when unset, the bridge auto-detects Python 3.12 from the
PEP 514 registry (`HKCU`/`HKLM\Software\Python\PythonCore\3.12\InstallPath`) and uses the result only if
`python312.dll` is actually present there. It does not depend on `pyvenv.cfg`. Set it explicitly only when
auto-detection fails. On Linux it is ignored.

**Windows (PowerShell)**:

```powershell
$env:HOOPS_AI_HOME = "<HOOPS_AI_HOME>"
```

**Linux**:

```bash
export HOOPS_AI_HOME=/path/to/HOOPS_AI/V1.1
export DISPLAY=:99      # headless only; Xvfb must already be running
```

> **Windows pitfall**: PowerShell has its own `set` (an alias of `Set-Variable`), so
> `set HOOPS_AI_HOME=...` silently creates a PowerShell variable instead of an environment variable, and the
> child process then fails with a confusing `ModuleNotFoundError: No module named 'hoops_ai'`. Use
> `$env:NAME = "value"` in PowerShell and `set NAME=value` in `cmd.exe`. Variables set with `setx` or through
> the GUI do not reach shells that were already open.

### Subcommands

Run with no arguments to print usage.

| Subcommand | Arguments | What it does |
|---|---|---|
| `mfr` | `<cad_file> <mfr_ckpt>` | MFR inference (per-face feature classification) for one CAD file |
| `embed` | `<cad_file> <embed_ckpt>` | Computes and prints the shape embedding vector for one CAD file |
| `compare` | `<cad_file_1> <cad_file_2> <embed_ckpt>` | Cosine similarity between two CAD files |
| `index-add` | `<cad_file> <embed_ckpt> [--index <path>]` | Opens the index (creating it if missing) and registers the file. A thumbnail is rendered during embedding; its path and whether it exists are printed |
| `index-search` | `<cad_file> <K> <embed_ckpt> [--index <path>]` | Top-K similar **parts**; prints each hit's id, score, and thumbnail path |
| `similar-assembly` | `<cad_file> <K> [<embed_ckpt>] [--index <path>]` | Ranks whole **assemblies** by similarity to the query. The checkpoint is needed only when the query is not already in the index (it is then embedded live); an in-corpus query reuses stored vectors. The first search on an index builds rarity weights — a few seconds on a large corpus — and caches them |
| `index-info` | `[--index <path>]` | Opens an existing index (never creates one, no model needed) and prints its path, dimension, and file / body / assembly / single-part counts |
| `index-close` | `[--index <path>]` | Opens then closes the index to verify close. Files are left on disk |

Checkpoints live under `<HOOPS_AI_HOME>/packages/trained_ml_models/`: `ts3d_162k_mfr.ckpt` for `<mfr_ckpt>`
and `ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt` for `<embed_ckpt>`.

**Windows (PowerShell)**:

```powershell
bin\win64\test_client.exe mfr          "<cad_file>" "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_162k_mfr.ckpt"
bin\win64\test_client.exe index-add    "<cad_file>" "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" --index C:\temp\my-index
bin\win64\test_client.exe index-search "<cad_file>" 5 "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" --index C:\temp\my-index
bin\win64\test_client.exe index-info   --index C:\temp\my-index
```

**Linux**:

```bash
./bin/linux64/test_client mfr          ./sample.step "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_162k_mfr.ckpt"
./bin/linux64/test_client index-add    ./sample.step "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" --index /tmp/my-index
./bin/linux64/test_client index-search ./sample.step 5 "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" --index /tmp/my-index
./bin/linux64/test_client index-info   --index /tmp/my-index
```

### How an index is stored

**The client owns the index location and name; the bridge has no default.** The path you pass is treated as
a *base path*: a trailing `.faiss` extension is stripped, anything else is used as-is. Given base
`<dir>/<name>`, the bridge maintains

```text
<dir>/<name>.faiss     FAISS vectors
<dir>/<name>.meta      per-record metadata
<dir>/<name>/          per-index folder holding the thumbnails hoops_ai renders during embedding,
                       laid out as <parent folder of the CAD file>/<file stem>_white.png
                       (each with a sibling .scs)
```

Saves are atomic (written to a temporary base, then renamed). Several indexes can coexist on disk;
`HoopsAI_OpenIndex` makes one of them current. The index dimension comes from the embeddings model, so load
the model before creating a new index. When `--index` is omitted, `test_client` falls back to base
`my_index` next to the executable — a **sample-side convenience only**.

Resolve a hit's thumbnail with `HoopsAI_GetPartThumbnailPath` rather than constructing the path yourself; it
prefers the path recorded in the part's metadata and also falls back to older flat `<stem>.png` names, so
indexes created by earlier versions or by HOOPS_AI-WebAPI still resolve.

To delete an index, run `index-close` first, then remove `<name>.faiss`, `<name>.meta` and the `<name>/`
folder yourself. The bridge never deletes index files — and if you delete them while the index is still
open, the next registration's save recreates them.

## Preparing a redistribution package

This is the scenario where a partner ships its own product — an application built on this bridge — to end
users who do not have HOOPS AI installed:

1. Build (above).
2. **Trace** which Python modules the features you actually use import (`tools/trace_modules.py`).
3. **Assemble** a staging directory from the trace result, the build output, the checkpoints, and a README
   (`tools/build_redist_package.ps1` / `.sh`).
4. Copy that directory to another machine and verify it there.

**Zipping is intentionally left out.** This repository produces only the bridge and the test client, not
your actual application, so the ZIP should be created once, together with your app, when the product itself
is packaged.

### Step 1: trace the modules you actually use

Which modules are needed depends entirely on which features you use. **Static metadata (`Requires-Dist` and
friends) overestimates**: it pulls in packages that are irrelevant at runtime, such as `scikit-learn`, which
is only used for training. The reliable method is to run the feature and diff `sys.modules`.

**Windows (PowerShell)**:

```powershell
$env:HOOPS_AI_HOME = "<HOOPS_AI_HOME>"
$env:HOOPS_AI_LICENSE = '...'   # single quotes: the key may contain '$'

python tools\trace_modules.py `
    --cad-file .\sample.step `
    --mfr-ckpt "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_162k_mfr.ckpt" `
    --embed-ckpt "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" `
    --out C:\temp\used_top_level_modules.txt
```

**Linux**:

```bash
export HOOPS_AI_HOME=/path/to/HOOPS_AI/V1.1
export HOOPS_AI_LICENSE='...'
export DISPLAY=:99   # headless only

python3 tools/trace_modules.py \
    --cad-file ./sample.step \
    --mfr-ckpt "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_162k_mfr.ckpt" \
    --embed-ckpt "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" \
    --out /tmp/used_top_level_modules.txt
```

`trace_modules.py` is a plain Python script, so it reads the license from `HOOPS_AI_LICENSE` itself (the
bridge does not), and derives `site-packages` from `HOOPS_AI_HOME`.

Its `run_traced_workload()` exercises MFR, shape embeddings, similar-part indexing (`FaissVectorStore`) and
thumbnail generation. **If your product uses a different feature set, edit that function to match before
running it — and re-run the trace whenever your feature set changes.**

### Step 2: assemble the staging directory

**Windows (PowerShell)**:

```powershell
$env:HOOPS_AI_HOME = "<HOOPS_AI_HOME>"

.\tools\build_redist_package.ps1 `
    -ModulesFile C:\temp\used_top_level_modules.txt `
    -Ckpt "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_162k_mfr.ckpt", "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" `
    -Sample C:\temp\sample.step `
    -OutDir C:\temp\output_dir
```

**Linux**:

```bash
export HOOPS_AI_HOME=/path/to/HOOPS_AI/V1.1

./tools/build_redist_package.sh \
    --modules-file /tmp/used_top_level_modules.txt \
    --ckpt "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_162k_mfr.ckpt" \
    --ckpt "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" \
    --sample ./sample.step \
    --out-dir /tmp/output_dir
```

Both scripts collect the build output, the traced subset of `site-packages` (bundled as `.venv`), the
checkpoints and samples you name, and `tools/redist_README.md.example` (copied as `README.md`) into
`<out dir>/redist_package/`. Add your product name and support contact to that README before shipping.

They also generate `THIRD_PARTY_MANIFEST.txt`, which lists every bundled package as either **TechSoft3D**
(redistribution has to be confirmed individually — see *Licensing and redistribution rights* below) or
**ThirdParty** (OSS and similar), so you can tell the HOOPS AI core apart from its dependencies.

> **Watch disk space**: the staging directory runs to several GB. Check free space at the output location
> first.

### Step 3: verify on another machine

Copy `redist_package/` over — a plain folder copy, or a throwaway ZIP just for the transfer; either way this
is not the final distribution artifact. Then `cd` into it and run the features your product uses. Note that
inside the package the binaries sit directly in `bin/`, not in `bin/win64/` or `bin/linux64/`.

**Windows (cmd.exe)** — in PowerShell, invoke the exe with a leading `.\`:

```bat
cd redist_package
bin\test_client.exe mfr   samples\<sample>.step models\ts3d_162k_mfr.ckpt
bin\test_client.exe embed samples\<sample>.step models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt
```

**Linux**:

```bash
cd redist_package
export DISPLAY=:99      # headless only; Xvfb must already be running
./bin/test_client mfr   samples/<sample>.step models/ts3d_162k_mfr.ckpt
./bin/test_client embed samples/<sample>.step models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt
```

`HOOPS_AI_HOME` does **not** need to be set here: the bundled `site-packages` sits next to `bin/` as
`.venv` and the client auto-detects it relative to its own location. The bridge library sits next to the
executable — found via the `$ORIGIN` rpath on Linux, so no `LD_LIBRARY_PATH`. No launcher wrapper is needed.

`[OK] HOOPS AI License: Valid` followed by inference output means the package is good.

> **`cd` into `redist_package` before running.** `GraphNodeClassification`, used internally by MFR, writes
> `ml_metrics/` and similar output to the current directory.

What the target machine needs in advance:

- The same CPU architecture.
- A **Python 3.12 runtime** — `python312.dll` / `libpython3.12.so.1.0` is *not* bundled. On Windows, install
  it with the python.org installer so it registers in the PEP 514 registry and the bridge finds it
  automatically.
- On headless Linux, Xvfb and OpenGL packages, installed as OS packages.
- For GPU inference, a `torch` build matching the machine's CUDA version. This repository verifies CPU only.

> `test_client` is a verification harness for whoever assembles the package, not something end users are
> meant to run. In a real product it is replaced by your own application UI.

### Size expectations

Measured with MFR + shape embeddings, `hoops_ai` 1.1.0, CPU `torch`. The trace came out at roughly 70–85
top-level packages, varying slightly from run to run.

| | Windows | Linux |
|---|---|---|
| Staging directory | ≈2.6 GB | ≈6.0 GB |
| The same content, ZIP-compressed | ≈1 GB | ≈2 GB |

`hoops_exchange`, `hoops_converter` and `torch` dominate, followed by `scipy`, `pyarrow`, `sympy` and
`pandas`; narrowing the feature set changes little unless one of the big three becomes unnecessary. Linux is
roughly twice the size of Windows — not because of CUDA (this is the CPU build of `torch`) but because the
HOOPS native libraries are larger there. Factor that in when estimating a download size for a Linux
product.

### Obtaining the packages

The approach here copies the traced packages straight out of a working development venv. `pip download` is
an alternative for the OSS dependencies, but the HOOPS AI core (`hoops_ai`, `hoops_exchange`,
`hoops_converter`, ...) is only available through Tech Soft 3D distribution channels, so copying from a venv
you have already verified is usually the most reliable route. Dependency licenses are listed on the
[Acknowledgments page](https://docs.techsoft3d.com/hoops/ai/resources/Acknowledgments.html); check whether
each package requires its LICENSE file to be included.

## Implementation notes

- **Similar-part search uses the low-level `FaissVectorStore`, not the high-level `CADSearch`**, following
  the official
  [Similarity Search Workflow](https://docs.techsoft3d.com/hoops/ai/programming_guide/embeddings-production.html)
  ("Using a Custom Vector Store"). This keeps the client free to hold several indexes on disk and switch the
  current one.
- **Two embedding paths, by design.** `embed_shape()` returns a **list of `Embedding` objects, one per
  body** — a part is not necessarily a single body, and assuming a single `Embedding` and reading `.values`
  raises `AttributeError: 'list' object has no attribute 'values'`. The bridge uses this two ways:
  - **Averaged single vector** — `HoopsAI_ComputeEmbedding` / `HoopsAI_CompareEmbeddings` L2-normalize each
    body vector, average them, and re-normalize into **one vector per file** (`ComputeEmbeddingVector` in
    `src/hoops_ai_bridge.cpp`). Simple for 1-to-1 comparison, at the cost of losing which parts are present
    and how many.
  - **Per-body index** — `HoopsAI_AddCADToIndex` / `HoopsAI_AddCADFolderToIndex` call `embed_shape_batch()`
    and store **one row per body** (rows of the same file share its path as their id), normalized but not
    averaged. This body-level layout is what makes both fine-grained similar-part search and
    assembly-to-assembly retrieval possible on one index.
- **Assembly search runs on the per-body index**, so build the index with the batch add functions before
  calling `HoopsAI_SearchSimilarAssembly`.
- **Batch worker count.** For `HoopsAI_AddCADFolderToIndex`, `numWorkers <= 0` lets the *bridge* choose a
  bounded worker count, and that is the recommended default. It deliberately does not delegate to
  `hoops_ai`'s all-logical-core auto-detect: every spawned worker keeps its **own** copy of the embedding
  model plus the working set of the file it is embedding, so peak RSS grows roughly linearly with the worker
  count. The useful worker count is therefore bounded by **both** the physical core count and available RAM,
  and the throughput-vs-workers curve rises to a plateau and then flattens (and, once RAM is the bottleneck,
  can actually *decline*).

  Where that plateau sits depends on the machine and, importantly, on how heavy the CAD files are, so treat
  the numbers below as illustrations of the *shape* of the curve rather than a fixed target:
    - **Light, mostly single-body parts** let the plateau extend up to about the physical core count. A local
      benchmark (14 physical / 20 logical cores, 31.7 GB, `parts500`) peaked around `num_workers = 12` with
      every setting from ~8 up to the core count within ~5 % of peak and no failures up to 18 workers.
    - **Heavy assemblies** cost more RAM and I/O per worker, so the plateau sits *below* the core count and
      over-subscribing hurts. An AWS `g6.8xlarge` run (16 physical / 32 logical cores, 121 GB, heavy
      `mechcad`) also peaked at `num_workers = 12`, but 16/20/24 workers were progressively **slower** than
      even 8 (24 workers were the slowest of all) while RSS climbed toward ~37 GB.

  Practical guidance that scales from an 8-core laptop to a many-core server: the sweet spot is roughly the
  physical core count for light parts, and a bit lower (order of ~8–12) for heavy assemblies; adding workers
  far beyond that only trades RAM for no gain, and can be slower. The exact peak is run-to-run noise, so aim
  for that plateau rather than a single best value. The simplest choice is to pass `numWorkers <= 0` and let
  the bridge bound the count (it uses the minimum of a cap, half the logical cores, and available-RAM /
  per-worker-model-footprint); on a high-core machine with plenty of RAM and light files, raise
  `HOOPS_AI_MAX_WORKERS` to let it climb toward the core count. All limits are tunable at runtime via
  `HOOPS_AI_MAX_WORKERS`, `HOOPS_AI_MODEL_FOOTPRINT_MB` and `HOOPS_AI_MIN_FILES_PARALLEL`.
- **Per-file time budget.** `HoopsAI_AddCADFolderToIndex` takes a `timeLimitSeconds` argument that is passed
  to `embed_shape_batch` as the per-file size-bucket limits (`time_limit_small/medium/large`; `<= 0` keeps
  `hoops_ai`'s 120 s default). A heavy assembly that exceeds the budget is dropped with a *Timeout* and
  returned in `outFailedPaths`, which enables a **two-pass add**: pass 1 embeds the whole folder at the
  default budget (the many light files succeed), then pass 2 re-adds only the failed paths with a larger
  `timeLimitSeconds` and a small `numWorkers` (a single heavy file is not sped up by workers; only
  across-file parallelism helps). `time_limit_overall` is intentionally not set, to avoid imposing a
  whole-batch cap.
- **Model loading is cached.** `HoopsAI_LoadMFRModel` / `HoopsAI_LoadEmbeddingsModel` construct the model
  and run `load_from_checkpoint` on the first call, then reuse it for the lifetime of the process.
- **First import is slow.** Loading PyTorch takes a while; run it on a worker thread so it does not block
  your UI.
- **Everything is serialized.** Every export is protected by a process-wide lock plus the GIL, so concurrent
  calls from multiple windows or documents queue rather than run in parallel. Thumbnail rendering can take
  several seconds per part, so keep add and search off the UI thread.

### Known limitations

- The averaged 1-to-1 path is lossy for multi-body parts; for rigorous multi-body or assembly comparison,
  use the per-body index plus `HoopsAI_SearchSimilarAssembly` instead.
- `AssemblyMatcher` is embedded tutorial source (Pattern B) and has to be re-synced with upstream by hand.
  It also reads private `CADSearch` internals (`_shape_model`, `_get_shape_embedding_batch()`) as a
  temporary workaround, pending public accessors upstream.
- A checkpoint cannot be swapped once loaded; switching requires `HoopsAI_Shutdown` and a process restart.
- GPU inference is unverified.

## Notes for productization

`include/hoops_ai_bridge.h` carries the same warning, and it bears repeating: the exported functions are
**not** meant to become your product's public API or an end-user-facing plugin interface. At minimum, an
application layer built on this should

- fix and bundle the checkpoint path at build or install time, rather than exposing it as a runtime
  parameter that accepts arbitrary paths;
- interpret raw MFR label IDs in your own business logic ("chamfer", "hole", ...) before showing anything to
  an end user;
- name and shape the exported functions around your product's features instead of HOOPS AI's internal
  concepts.

## Licensing and redistribution rights

Everything above concerns what works technically. **Whether you may ship it is a separate question.**
Whether Tech Soft 3D packages (`hoops_ai`, `hoops_exchange`, `hoops_converter`, `hoops_web_viewer`, ...),
pretrained checkpoints, or a license key embedded in your binary may go out to end users — and under what
conditions — is determined by the terms of the **TPA (Technology Partner Agreement) between your company and
Tech Soft 3D**, and can differ by country and contract.

Don't treat this repository as the answer. Confirm with your own licensing contact and with Tech Soft 3D.
The verification here establishes only that this works technically, not that it is permitted.
