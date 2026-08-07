// hoops_ai_bridge.cpp
// Embeds CPython and exposes MFR inference, Shape Embeddings similarity, a similar-parts
// FAISS index, and assembly-to-assembly search from hoops_ai (Nuitka-compiled .pyd/.so)
// as C ABI functions.
//
// Referenced Python-side call patterns:
//   MFR (V1.1 notebooks 5a and 4c):
//     from hoops_ai.cadaccess import HOOPSLoader
//     from hoops_ai.ml.EXPERIMENTAL import FlowInference, GraphNodeClassification
//     loader = HOOPSLoader()
//     model  = FlowInference(cad_loader=loader,
//                            flowmodel=GraphNodeClassification(result_dir=out_dir))
//     model.load_from_checkpoint(checkpoint_path)
//     ml_input = model.preprocess(cad_file_path)
//     predictions, probabilities = model.predict_and_postprocess(ml_input)
//
//   Embeddings (HOOPS_AI-WebAPI: get_embedder/compute_embedding/compare_embeddings in core.py):
//     from hoops_ai.ml.embeddings import HOOPSEmbeddings
//     HOOPSEmbeddings.register_model(model_name="hoops_embeddings_signal",
//                                    checkpoint_path=checkpoint_path)
//     embedder = HOOPSEmbeddings(cad_loader=loader, model="hoops_embeddings_signal")
//     embedding = embedder.embed_shape(cad_file_path)   # Embedding(values: np.ndarray shape (dim,))
//     # Similarity is computed by calling compute_embedding twice, applying L2 normalization,
//     # and then comparing with the dot product (cosine similarity)
//     # (For the simple 1-to-1 comparison and the similar-parts index, hoops_ai.ml.CADSearch is
//     #  not used -- it is overkill; the index goes straight to the low-level FaissVectorStore.
//     #  CADSearch IS used, however, by the assembly-to-assembly search below, whose AssemblyMatcher
//     #  needs its shape model and stored corpus.)
//
//   Similar Parts Index (refer to create_index/add_to_index/search_index/list_indexes
//   in HOOPS_AI-WebAPI core.py. The client chooses each index's base path; HoopsAI_OpenIndex
//   opens one and makes it the current index (opening a different path switches which one is
//   current), while multiple indexes may live on disk):
//     from hoops_ai.ml.embeddings import FaissVectorStore, Embedding, VectorRecord
//     vs = FaissVectorStore(dim)                  # Create a new empty index
//     vs.save(base)                                # Save .faiss / .meta to base (without extension)
//     vs = FaissVectorStore.load(base)             # Load an existing index (returns a new instance)
//     rec = VectorRecord(id=part_id, embedding=Embedding(values=vec, model=..., dim=...),
//                        metadata={"file_id": part_id, "filename": ...})
//     if part_id in vs.get_ids(): vs.delete([part_id])  # Prevent duplicates during re-registration
//     vs.upsert([rec]); vs.save(base)
//     hits = vs.query(vector, top_k=k)             # hit.id / hit.score / hit.metadata
//   NOTE: registration (HoopsAI_AddCADToIndex / ...AddCADFolderToIndex) actually embeds via
//   embedder.embed_shape_batch(...) and upserts ONE VectorRecord per body (multiple rows share
//   the file path as id), instead of the single averaged vector shown above. This per-body layout
//   is what enables fine-grained similar-part search and the assembly search below.
//
//   Assembly-to-assembly search (HoopsAI_SearchSimilarAssembly):
//     Built on the per-body index above via AssemblyMatcher, which is NOT part of the compiled
//     hoops_ai package but tutorial source embedded from src/assembly_matcher_py.h and exec'd once
//     (see EnsureAssemblyModule / kAssemblyMatcherPy). It performs part-level optimal 1-to-1 matching
//     (Hungarian) + TF-IDF rare-part weighting + bag-of-parts composition scoring.
//
// CAD ACCESS (HOOPSLoader creation) is required by both MFR and Embeddings,
// so it is shared through GetSharedCADLoader() (the HOOPSEmbeddings constructor can also
// optionally accept cad_loader: hoops_ai.ml.embeddings.HOOPSEmbeddings([cad_loader, model, device])).
//
// [Important design note] (see hoops_ai_bridge.h for details)
// This bridge is a reference implementation showing the HOOPS AI invocation mechanism.
// It is intended as a foundation for a partner-provided value-added module, not as a simple wrapper
// around HOOPS AI tutorial features and samples (for example, building and running inference with an
// end-user-specific trained model based on the company's own CAD history data, or connecting to an
// internal knowledge base to predict cost or machining processes). The functions implemented here,
// such as RunMFRInference and LoadEmbeddingsModel, should not be exposed directly as a public API
// for end users.

#define HOOPSAI_BRIDGE_EXPORTS
#include "hoops_ai_bridge.h"
#include "assembly_matcher_py.h"

#include <Python.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cctype>
#include <ctime>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <random>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace {

namespace fs = std::filesystem;

std::mutex g_pyMutex;
bool g_initialized = false;

// True once HoopsAI_Initialize has pointed Python's multiprocessing at the real python.exe
// (see ConfigureMultiprocessingSpawn). While false, the batch add code clamps num_workers to 1
// because spawning workers off the HOST executable (this embedded interpreter's sys.executable)
// would relaunch the host application. When true, num_workers > 1 and the None/auto-detect
// default are safe.
bool g_mpConfigured = false;

// Main-thread Python state saved by PyEval_SaveThread() at the end of HoopsAI_Initialize.
// Releasing the GIL after start-up is what lets the exported functions be driven from a
// worker thread (their PyGILState_Ensure/Release re-acquires it). Restored in HoopsAI_Shutdown
// right before Py_FinalizeEx. nullptr while uninitialized.
PyThreadState* g_mainThreadState = nullptr;

// CAD ACCESS: HOOPSLoader instance shared by MFR and Embeddings (lazy-created and reused).
PyObject* g_sharedCadLoader = nullptr;

// Embeddings: loaded HOOPSEmbeddings instance (created by HoopsAI_LoadEmbeddingsModel).
PyObject* g_embedder = nullptr;
const char* kEmbeddingsModelName = "hoops_embeddings_signal";
// Checkpoint path currently loaded into g_embedder (empty until the first load).
// Used to decide "reuse vs. reload" when a different checkpoint is requested.
std::string g_embedderCkptPath;
// HOOPSEmbeddings.register_model rejects re-registering the same model_name and
// there is no public unregister API, so every (re)load allocates a fresh unique
// model_name. g_embModelSeq is the counter and g_currentEmbModelName holds the
// name backing the live g_embedder (also stamped onto index records).
int g_embModelSeq = 0;
std::string g_currentEmbModelName = kEmbeddingsModelName;

// Folder-add progress forwarding (see HoopsAI_SetProgressCallback). g_progressCb/g_progressUser
// are set under g_pyMutex and read on the worker thread while a folder add runs. g_progressShim
// caches the Python stderr-shim CLASS object (built lazily via kProgressShimPy) that parses the
// tqdm bar and calls back into the bridge. All three are only touched with the GIL held except
// the plain-pointer store in HoopsAI_SetProgressCallback.
HoopsAI_ProgressCallback g_progressCb = nullptr;
void* g_progressUser = nullptr;
PyObject* g_progressShimClass = nullptr;

// MFR: loaded FlowInference instance (created by HoopsAI_LoadMFRModel).
PyObject* g_mfrModel = nullptr;
// Checkpoint path currently loaded into g_mfrModel (empty until the first load).
std::string g_mfrCkptPath;

// Similar Parts Index: FAISS index for similar-part search. HoopsAI_OpenIndex opens one
// and makes it the current index; the client decides the base path and the bridge holds
// no default. Nothing is prepared in HoopsAI_Initialize.
PyObject* g_index = nullptr;
// Assembly-search namespace: a private dict into which the embedded AssemblyMatcher source
// (kAssemblyMatcherPy) plus its bridge driver (kAssemblyDriverPy) are exec'd exactly once,
// lazily on the first assembly search / index-stats call. Holds the per-index matcher cache
// (keyed by faiss path + mtime) so the ~seconds-long IDF build is paid only once per index.
// Kept alive for the process lifetime (never DECREF'd); cleared/rebuilt is unnecessary because
// the cache self-invalidates on file mtime change.
PyObject* g_asmNs = nullptr;
// Shape-map namespace: a private dict into which the embedded shape-map driver (kShapeMapPy) is
// exec'd exactly once, lazily on the first HoopsAI_ComputeIndexShapeMap call. Kept alive for the
// process lifetime (never DECREF'd). Independent of g_asmNs (no AssemblyMatcher dependency).
PyObject* g_shapeMapNs = nullptr;
// Fallback dimension used ONLY when the embeddings model dimension cannot be obtained
// (see DeriveEmbeddingDim). If you ever see a new index created at this size unexpectedly,
// the HOOPSEmbeddings dim attribute names below likely changed and need updating.
constexpr int kIndexDim = 2048;
// Dimension of the current index (0 when there is no current index).
int g_indexDim = 0;
// Base path of the current index (<dir>/<stem>, WITHOUT extension), derived from the .faiss
// file the client opened; empty when there is no current index. The bridge uses
// <g_indexBase>.faiss / <g_indexBase>.meta and the per-index folder <g_indexBase>/<stem>.png
// (a folder named after the index) inside it.
std::string g_indexBase;

void SetError(char* outErrorMsg, int errorMsgSize, const std::string& msg) {
    if (!outErrorMsg || errorMsgSize <= 0) return;
    std::strncpy(outErrorMsg, msg.c_str(), errorMsgSize - 1);
    outErrorMsg[errorMsgSize - 1] = '\0';
}

// --- Index path layout helpers (file-based index; see hoops_ai_bridge.h) ---
// The client opens/creates a .faiss FILE (its parent folder name is free). From that path:
//   base = <dir>/<stem>   (path WITHOUT extension, handed to FaissVectorStore.save/load)
//   <base>.faiss          FAISS vectors (the file the client selected)
//   <base>.meta           per-record metadata
//   <base>/               per-index folder (named after the index); holds thumbnails

// Replace '\\' with '/' so paths are uniform in JSON / Qt and comparable as strings.
std::string NormalizeSlashes(std::string p) {
    for (char& c : p) if (c == '\\') c = '/';
    return p;
}

// Build a std::filesystem::path from a UTF-8 string across platforms.
fs::path U8Path(const std::string& s) {
    return fs::u8path(s);
}

// Return a std::error_code's message as UTF-8. On Windows std::error_code::message()
// yields the OS message in the local ANSI code page (e.g. CP932), which becomes mojibake
// when handed back to a UTF-8 client; re-encode it to UTF-8 here. Elsewhere the message is
// already UTF-8.
std::string SysErrUtf8(const std::error_code& ec) {
#if defined(_WIN32)
    const std::string ansi = ec.message();
    if (ansi.empty()) return ansi;
    int wlen = MultiByteToWideChar(CP_ACP, 0, ansi.c_str(), (int)ansi.size(), nullptr, 0);
    if (wlen <= 0) return ansi;
    std::wstring wide(wlen, L'\0');
    MultiByteToWideChar(CP_ACP, 0, ansi.c_str(), (int)ansi.size(), &wide[0], wlen);
    int ulen = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), wlen, nullptr, 0, nullptr, nullptr);
    if (ulen <= 0) return ansi;
    std::string utf8(ulen, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), wlen, &utf8[0], ulen, nullptr, nullptr);
    return utf8;
#else
    return ec.message();
#endif
}

// Ensure that a directory exists, creating it (recursively) when missing. Unlike a bare
// fs::create_directories call this tolerates the common cases gracefully:
//   - an empty path (e.g. a relative leaf that has no parent) is a no-op success;
//   - an already-existing directory is success (some std::filesystem implementations report
//     a spurious "already exists" error_code when the leaf is present);
//   - an existing NON-directory (a file sits where the folder should go) is a clear error.
// On failure returns false with a readable, UTF-8 message in errOut.
bool EnsureDir(const fs::path& dir, std::string& errOut) {
    if (dir.empty()) return true;
    std::error_code ec;
    if (fs::exists(dir, ec)) {
        if (fs::is_directory(dir, ec)) return true;
        errOut = "a file already exists where a folder is required: " +
                 NormalizeSlashes(dir.u8string());
        return false;
    }
    ec.clear();
    fs::create_directories(dir, ec);
    if (ec) {
        errOut = "create directory failed (" + NormalizeSlashes(dir.u8string()) + "): " +
                 SysErrUtf8(ec);
        return false;
    }
    return true;
}


// path (<dir>/<stem>). A path with any other/no extension is returned unchanged.
std::string IndexBaseFromFaiss(const std::string& faissPath) {
    fs::path p = U8Path(faissPath);
    if (p.extension() == fs::path(".faiss")) p.replace_extension();
    return NormalizeSlashes(p.u8string());
}
// <base>.faiss  (used to detect an existing index).
std::string IndexFaissOf(const std::string& base) {
    return NormalizeSlashes(base + ".faiss");
}
// <base>/  : the per-index folder (named after the index); thumbnails are created here
// lazily on the first add.
std::string ThumbnailsDirOf(const std::string& base) {
    return NormalizeSlashes(base);
}

// Short random hex token for temp file names (atomic save / thumbnail scratch).
std::string RandomHex(int nbytes) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 15);
    static const char kHex[] = "0123456789abcdef";
    std::string s;
    s.reserve(static_cast<size_t>(nbytes) * 2);
    for (int i = 0; i < nbytes * 2; ++i) s += kHex[dist(rng)];
    return s;
}

// Current UTC time formatted as "%Y-%m-%dT%H:%M:%SZ" (matches Web API core.py registered_at).
std::string UtcTimestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    gmtime_s(&tmv, &t);
#else
    gmtime_r(&t, &tmv);
#endif
    char buf[32] = {0};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return buf;
}

#if defined(_WIN32)
// Convert a narrow (const char*, ANSI/CP_ACP-encoded, as returned by getenv or
// passed through the C ABI) string to UTF-16. Using MultiByteToWideChar instead
// of a byte-wise widening ensures non-ASCII paths (e.g. a Japanese Windows user
// name such as C:\Users\山田\...) are decoded correctly.
std::wstring NarrowToWide(const char* narrow) {
    if (!narrow || !*narrow) return std::wstring();
    int len = MultiByteToWideChar(CP_ACP, 0, narrow, -1, nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring wide(static_cast<size_t>(len), L'\0');
    int written = MultiByteToWideChar(CP_ACP, 0, narrow, -1, &wide[0], len);
    if (written <= 0) return std::wstring();
    // Drop the terminating NUL that MultiByteToWideChar counted/wrote.
    if (!wide.empty() && wide.back() == L'\0') wide.pop_back();
    return wide;
}

// Confirm that <home>\python312.dll actually exists under the candidate
// directory. Trailing backslashes are trimmed first so both "C:\Python312"
// and "C:\Python312\" are accepted. Uses the wide-character file API so
// non-ASCII paths are handled correctly.
bool HasPython312Dll(std::wstring home) {
    while (!home.empty() && (home.back() == L'\\' || home.back() == L'/')) {
        home.pop_back();
    }
    if (home.empty()) return false;
    std::wstring dll = home + L"\\python312.dll";
    DWORD attr = GetFileAttributesW(dll.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) &&
           !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

// Read the default value (REG_SZ) of a PEP 514 InstallPath key, e.g.
// HKCU/HKLM\Software\Python\PythonCore\3.12\InstallPath. Uses the wide-character
// registry API so non-ASCII install paths survive unmangled.
std::wstring ReadRegistryInstallPath(HKEY rootKey) {
    HKEY hKey;
    if (RegOpenKeyExW(rootKey,
                      L"Software\\Python\\PythonCore\\3.12\\InstallPath",
                      0, KEY_READ, &hKey) != ERROR_SUCCESS) {
        return std::wstring();
    }
    wchar_t buffer[MAX_PATH] = {0};
    DWORD type = 0;
    DWORD size = sizeof(buffer); // size in bytes
    LONG rc = RegQueryValueExW(hKey, nullptr, nullptr, &type,
                               reinterpret_cast<LPBYTE>(buffer), &size);
    RegCloseKey(hKey);
    if (rc != ERROR_SUCCESS || type != REG_SZ) return std::wstring();
    // REG_SZ is not guaranteed to be NUL-terminated; bound by the returned
    // byte count (converted to wchar_t count) and NUL-terminate defensively.
    size_t chars = size / sizeof(wchar_t);
    if (chars >= MAX_PATH) chars = MAX_PATH - 1;
    buffer[chars] = L'\0';
    return std::wstring(buffer);
}

// Resolve the Python 3.12 home directory (the base interpreter that provides
// python312.dll and the standard library) on Windows, in priority order:
//   1. The explicit argPythonHome argument (highest-priority override).
//   2. The HOOPS_AI_PYTHON_HOME environment variable (compat / escape hatch).
//   3. The PEP 514 registry InstallPath for CPython 3.12 (HKCU, then HKLM).
// Each candidate is only accepted if <home>\python312.dll actually exists.
// Returns an empty wstring if none could be resolved (the caller then skips
// Py_SetPythonHome and lets the embedded interpreter fall back to its own
// default search). All processing is done in UTF-16 so non-ASCII paths work.
std::wstring ResolvePythonHomeWin(const char* argPythonHome) {
    std::wstring argHome = NarrowToWide(argPythonHome);
    if (!argHome.empty() && HasPython312Dll(argHome)) {
        return argHome;
    }
    std::wstring envHome = NarrowToWide(std::getenv("HOOPS_AI_PYTHON_HOME"));
    if (!envHome.empty() && HasPython312Dll(envHome)) {
        return envHome;
    }
    std::wstring regHome = ReadRegistryInstallPath(HKEY_CURRENT_USER);
    if (!regHome.empty() && HasPython312Dll(regHome)) {
        return regHome;
    }
    regHome = ReadRegistryInstallPath(HKEY_LOCAL_MACHINE);
    if (!regHome.empty() && HasPython312Dll(regHome)) {
        return regHome;
    }
    return std::wstring();
}
#endif

// Using only str(value) may show only a generic C API boundary error such as
// "returned a result with an exception set", so use the traceback module to
// stringify the full stack trace.
std::string FetchPythonError() {
    if (!PyErr_Occurred()) return "unknown error";
    PyObject *type, *value, *tb;
    PyErr_Fetch(&type, &value, &tb);
    PyErr_NormalizeException(&type, &value, &tb);
    if (tb) PyException_SetTraceback(value, tb);

    std::string result;

    PyObject* tbModule = PyImport_ImportModule("traceback");
    if (tbModule) {
        PyObject* formatted = PyObject_CallMethod(tbModule, "format_exception", "(OOO)",
                                                   type, value, tb ? tb : Py_None);
        if (formatted) {
            PyObject* sep = PyUnicode_FromString("");
            PyObject* joined = PyUnicode_Join(sep, formatted);
            if (joined) {
                const char* utf8 = PyUnicode_AsUTF8(joined);
                if (utf8) result = utf8;
                Py_DECREF(joined);
            }
            Py_DECREF(sep);
            Py_DECREF(formatted);
        } else {
            PyErr_Clear();
        }
        Py_DECREF(tbModule);
    }

    if (result.empty() && value) {
        PyObject* str = PyObject_Str(value);
        if (str) {
            const char* utf8 = PyUnicode_AsUTF8(str);
            if (utf8) result = utf8;
            Py_DECREF(str);
        }
    }
    if (result.empty()) result = "python exception (could not stringify)";

    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(tb);
    return result;
}

// CAD ACCESS: create HOOPSLoader only once and reuse it for later calls.
// The caller must hold the GIL before calling this. The return value is a borrowed reference
// (the caller does not need to DECREF it).
PyObject* GetSharedCADLoader(std::string& errOut) {
    if (g_sharedCadLoader) return g_sharedCadLoader;

    PyObject* cadaccessMod = PyImport_ImportModule("hoops_ai.cadaccess");
    if (!cadaccessMod) {
        errOut = "import hoops_ai.cadaccess failed: " + FetchPythonError();
        return nullptr;
    }
    PyObject* loaderClass = PyObject_GetAttrString(cadaccessMod, "HOOPSLoader");
    Py_DECREF(cadaccessMod);
    if (!loaderClass) {
        errOut = "resolve HOOPSLoader failed: " + FetchPythonError();
        return nullptr;
    }
    PyObject* loader = PyObject_CallObject(loaderClass, nullptr);
    Py_DECREF(loaderClass);
    if (!loader) {
        errOut = "construct HOOPSLoader failed: " + FetchPythonError();
        return nullptr;
    }

    g_sharedCadLoader = loader; // Keep for the lifetime of the process (do not transfer the refcount)
    return g_sharedCadLoader;
}

// L2 normalization (same as core._l2_normalize in the Web API version: if the norm is nearly 0, return as-is).
void L2Normalize(std::vector<float>& v) {
    double norm = 0.0;
    for (float x : v) norm += static_cast<double>(x) * x;
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (float& x : v) x = static_cast<float>(x / norm);
    }
}

// Convert a Python numeric list / 1D NumPy array to std::vector<float>.
// If values itself is a NumPy array, call .tolist() on it before passing it in.
bool PyListToFloatVector(PyObject* pyList, std::vector<float>& out, std::string& errOut) {
    if (!PyList_Check(pyList) && !PyTuple_Check(pyList)) {
        errOut = "expected a list/tuple of floats";
        return false;
    }
    Py_ssize_t n = PySequence_Size(pyList);
    out.resize(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_GetItem(pyList, i); // New reference
        if (!item) { errOut = "PySequence_GetItem failed"; return false; }
        out[static_cast<size_t>(i)] = static_cast<float>(PyFloat_AsDouble(item));
        Py_DECREF(item);
    }
    return true;
}

// Similar Parts Index: return the current index, or nullptr with an error if there is
// no current index. This is a plain accessor: it never lazily creates or loads an index
// (that is the responsibility of HoopsAI_OpenIndex). The caller must hold the GIL.
// The return value is a borrowed reference (no DECREF needed).
PyObject* GetSharedIndex(std::string& errOut) {
    if (g_index) return g_index;
    errOut = "no current index: call HoopsAI_OpenIndex first";
    return nullptr;
}

// Release the current index from memory WITHOUT touching any files on disk. Idempotent.
// Used both by HoopsAI_CloseIndex and when the embeddings model is swapped (vectors from
// different models are not comparable and must not be mixed in one index). The caller must
// hold the GIL. Note: no std::remove here on purpose; index files are client-owned.
void CloseCurrentIndex() {
    Py_XDECREF(g_index);
    g_index = nullptr;
    g_indexBase.clear();
    g_indexDim = 0;
}

// Derive the embedding dimension from the currently loaded HOOPSEmbeddings instance.
// The attribute name is not guaranteed, so try a few candidates in order and use the first
// integer we can read. Falls back to the well-known SIGNAL default (kIndexDim = 2048) only
// when none are readable; if the fallback ever fires unexpectedly, the HOOPSEmbeddings API
// likely renamed these attributes and this list needs updating. The caller must hold the GIL
// and must have already loaded g_embedder.
int DeriveEmbeddingDim() {
    static const char* const kDimAttrs[] = {"dim", "embedding_dim", "output_dim"};
    for (const char* attr : kDimAttrs) {
        if (!g_embedder) break;
        PyObject* v = PyObject_GetAttrString(g_embedder, attr);
        if (!v) { PyErr_Clear(); continue; }
        if (PyLong_Check(v)) {
            long d = PyLong_AsLong(v);
            Py_DECREF(v);
            if (d > 0) return static_cast<int>(d);
            continue;
        }
        Py_DECREF(v);
    }
    // Fallback: attribute not exposed by this HOOPSEmbeddings version. Update kDimAttrs above
    // if this fires for a model whose real dimension differs from the SIGNAL default.
    return kIndexDim;
}

// Call embedder.embed_shape(cad_file_path) and aggregate the result into a single vector
// using the same logic as the Web API version (L2-normalize each body, then if there are
// multiple bodies, average them and apply L2 normalization again). Shared by
// HoopsAI_ComputeEmbedding / HoopsAI_CompareEmbeddings and by the QUERY side of
// HoopsAI_SearchIndex. NOTE: index REGISTRATION (HoopsAI_AddCADToIndex /
// HoopsAI_AddCADFolderToIndex) no longer uses this averaged path -- it embeds one row per
// body via embed_shape_batch (see AddPathsBatchCore) so multi-body files are stored per-body.
// The caller must hold the GIL and ensure g_embedder has already been loaded.
//
// Note: this averaging-based aggregation is not suitable for strict comparison of multi-body assemblies.
// The official embeddings_pipeline/demo_assembly_to_assembly_retrieval.ipynb performs assembly similarity
// search by separately indexing per-body vectors (multiple rows for the same file, sharing the file path as ID),
// then comparing assemblies through a multi-stage process: part-level optimal 1-to-1 matching using the
// Hungarian algorithm in AssemblyMatcher, rare-part weighting with TF-IDF, and bag-of-parts composition
// comparison. The "average all bodies into one vector" approach used here loses that composition information
// (which parts are included, how many there are, and the part-level correspondence) through averaging. It is
// adequate for comparing single-body parts, but should be understood as a simplified design choice that is not
// suitable for comparing or searching multi-body assemblies at accuracy equivalent to the official assembly search.
// For that rigorous multi-body case the bridge now provides HoopsAI_SearchSimilarAssembly, which runs the
// AssemblyMatcher pipeline above on the per-body index built by HoopsAI_AddCADToIndex / HoopsAI_AddCADFolderToIndex.
bool ComputeEmbeddingVector(const char* cadFilePath, std::vector<float>& outVec, std::string& errOut) {
    // embedder.embed_shape(cad_file_path) returns a list of embeddings, one for each body
    // (each solid that composes the part), not a single Embedding. This actual behavior is not
    // obvious from the class definitions alone; it became clear only by inspecting
    // HOOPS_AI-WebAPI core.py:1378-1395 (the compute_embedding implementation).
    // For multi-body CAD files (assemblies or STEP files with multiple solids), this becomes a list with multiple elements.
    PyObject* rawEmbeddings = PyObject_CallMethod(g_embedder, "embed_shape", "(s)", cadFilePath);
    if (!rawEmbeddings) {
        errOut = "embed_shape failed: " + FetchPythonError();
        return false;
    }

    PyObject* seq = PySequence_Fast(rawEmbeddings, "embed_shape() result is not a sequence");
    Py_DECREF(rawEmbeddings);
    if (!seq) {
        errOut = "embed_shape() result not iterable: " + FetchPythonError();
        return false;
    }

    Py_ssize_t numBodies = PySequence_Fast_GET_SIZE(seq);
    if (numBodies <= 0) {
        errOut = "embed_shape() returned no embeddings";
        Py_DECREF(seq);
        return false;
    }

    std::vector<std::vector<float>> bodyVectors;
    bool bodyOk = true;
    std::string bodyErr;
    for (Py_ssize_t i = 0; i < numBodies && bodyOk; ++i) {
        PyObject* emb = PySequence_Fast_GET_ITEM(seq, i); // borrowed

        // If this is an Embedding object, use .values; if it is a NumPy array, then also call .tolist().
        // Fall back so that even a raw array/list returned directly can still be handled.
        PyObject* valuesObj = PyObject_GetAttrString(emb, "values");
        if (!valuesObj) {
            PyErr_Clear();
            valuesObj = emb;
            Py_INCREF(valuesObj);
        }

        PyObject* asList = valuesObj;
        Py_INCREF(asList);
        if (PyObject_HasAttrString(valuesObj, "tolist")) {
            PyObject* converted = PyObject_CallMethod(valuesObj, "tolist", nullptr);
            if (converted) {
                Py_DECREF(asList);
                asList = converted;
            } else {
                PyErr_Clear();
            }
        }
        Py_DECREF(valuesObj);

        std::vector<float> vec;
        if (!PyListToFloatVector(asList, vec, bodyErr)) {
            bodyOk = false;
        }
        Py_DECREF(asList);
        if (!bodyOk) break;

        L2Normalize(vec);
        bodyVectors.push_back(std::move(vec));
    }
    Py_DECREF(seq);

    if (!bodyOk) {
        errOut = "parse body embedding failed: " + bodyErr;
        return false;
    }

    if (bodyVectors.size() == 1) {
        outVec = bodyVectors[0];
    } else {
        size_t dim = bodyVectors[0].size();
        outVec.assign(dim, 0.0f);
        for (auto& bv : bodyVectors) {
            for (size_t i = 0; i < dim && i < bv.size(); ++i) outVec[i] += bv[i];
        }
        for (float& x : outVec) x /= static_cast<float>(bodyVectors.size());
        L2Normalize(outVec); // Apply L2 normalization again after averaging (same as Web API version)
    }
    return true;
}

// Construct hoops_ai.ml.embeddings.Embedding / VectorRecord (modeled after add_to_index in the Web API version).
// thumbnailName / registeredAt are stored into the record metadata ("thumbnail" is the PNG
// file name only, no directory, so the index stays relocatable; "registered_at" is UTC).
// The caller must hold the GIL before calling. The return value is a new reference (the caller must DECREF it).
PyObject* BuildVectorRecord(const std::string& id, const std::string& filename,
                            const std::string& thumbnailName, const std::string& registeredAt,
                            const std::string& kind,
                            const std::vector<float>& vec, std::string& errOut) {
    PyObject* npMod = PyImport_ImportModule("numpy");
    if (!npMod) {
        errOut = "import numpy failed: " + FetchPythonError();
        return nullptr;
    }

    PyObject* embMod = PyImport_ImportModule("hoops_ai.ml.embeddings");
    if (!embMod) {
        Py_DECREF(npMod);
        errOut = "import hoops_ai.ml.embeddings failed: " + FetchPythonError();
        return nullptr;
    }
    PyObject* embeddingClass = PyObject_GetAttrString(embMod, "Embedding");
    PyObject* vectorRecordClass = PyObject_GetAttrString(embMod, "VectorRecord");
    Py_DECREF(embMod);
    if (!embeddingClass || !vectorRecordClass) {
        Py_XDECREF(embeddingClass);
        Py_XDECREF(vectorRecordClass);
        Py_DECREF(npMod);
        errOut = "resolve Embedding/VectorRecord failed: " + FetchPythonError();
        return nullptr;
    }

    // vec_np = numpy.asarray(<Python list version of vec>, dtype='float32')
    PyObject* pyList = PyList_New(static_cast<Py_ssize_t>(vec.size()));
    for (size_t i = 0; i < vec.size(); ++i) {
        PyList_SET_ITEM(pyList, static_cast<Py_ssize_t>(i), PyFloat_FromDouble(vec[i])); // Transfer the reference
    }
    PyObject* npArr = PyObject_CallMethod(npMod, "asarray", "(Os)", pyList, "float32");
    Py_DECREF(npMod);
    Py_DECREF(pyList);
    if (!npArr) {
        Py_DECREF(embeddingClass);
        Py_DECREF(vectorRecordClass);
        errOut = "numpy.asarray failed: " + FetchPythonError();
        return nullptr;
    }

    // emb = Embedding(values=vec_np, model="hoops_embeddings_signal", dim=<dim>)
    PyObject* kwargsEmb = Py_BuildValue("{s:O,s:s,s:i}",
                                        "values", npArr,
                                        "model", g_currentEmbModelName.c_str(),
                                        "dim", static_cast<int>(vec.size()));
    Py_DECREF(npArr);
    PyObject* embObj = PyObject_Call(embeddingClass, PyTuple_New(0), kwargsEmb);
    Py_DECREF(kwargsEmb);
    Py_DECREF(embeddingClass);
    if (!embObj) {
        Py_DECREF(vectorRecordClass);
        errOut = "construct Embedding failed: " + FetchPythonError();
        return nullptr;
    }

    // rec = VectorRecord(id=<partId or cadFilePath>, embedding=emb,
    //                    metadata={"file_id": <id>, "filename": <basename>,
    //                              "thumbnail": <stem>.png, "registered_at": <UTC>,
    //                              "kind": "part"|"assembly"})
    // "kind" (from embed_shape_batch metadata: single-body => "part", multi-body => "assembly")
    // lets part search exclude assemblies. Omitted from the dict when unknown so the record stays
    // compatible with the kind-less legacy layout.
    PyObject* metaDict = Py_BuildValue("{s:s,s:s,s:s,s:s}",
                                       "file_id", id.c_str(),
                                       "filename", filename.c_str(),
                                       "thumbnail", thumbnailName.c_str(),
                                       "registered_at", registeredAt.c_str());
    if (!kind.empty() && metaDict) {
        PyObject* kindVal = PyUnicode_FromString(kind.c_str());
        if (kindVal) {
            PyDict_SetItemString(metaDict, "kind", kindVal);
            Py_DECREF(kindVal);
        }
    }
    PyObject* kwargsRec = Py_BuildValue("{s:s,s:O,s:O}",
                                        "id", id.c_str(),
                                        "embedding", embObj,
                                        "metadata", metaDict);
    Py_DECREF(metaDict);
    Py_DECREF(embObj);
    PyObject* recObj = PyObject_Call(vectorRecordClass, PyTuple_New(0), kwargsRec);
    Py_DECREF(kwargsRec);
    Py_DECREF(vectorRecordClass);
    if (!recObj) {
        errOut = "construct VectorRecord failed: " + FetchPythonError();
        return nullptr;
    }
    return recObj;
}

// Compute the PNG stem for a part id (single, non-public site for the naming rule):
//   - if partId matches ^[A-Za-z0-9._-]{1,100}$ it is used verbatim;
//   - otherwise the lowercase hex SHA-256 of its UTF-8 bytes is used (via Python hashlib,
//     so no native crypto dependency is added).
// Web API part ids are already SHA-256 hex, so they take the first branch and produce
// <part_id>.png exactly like the Web API (interoperable). The caller must hold the GIL.
std::string ThumbnailStem(const std::string& partId) {
    bool safe = !partId.empty() && partId.size() <= 100;
    if (safe) {
        for (unsigned char c : partId) {
            if (!(std::isalnum(c) || c == '.' || c == '_' || c == '-')) { safe = false; break; }
        }
    }
    if (safe) return partId;

    std::string hex;
    PyObject* hashlib = PyImport_ImportModule("hashlib");
    if (hashlib) {
        PyObject* data = PyBytes_FromStringAndSize(partId.data(),
                                                   static_cast<Py_ssize_t>(partId.size()));
        if (data) {
            PyObject* h = PyObject_CallMethod(hashlib, "sha256", "(O)", data);
            Py_DECREF(data);
            if (h) {
                PyObject* hd = PyObject_CallMethod(h, "hexdigest", nullptr);
                Py_DECREF(h);
                if (hd) {
                    const char* u = PyUnicode_AsUTF8(hd);
                    if (u) hex = u;
                    Py_DECREF(hd);
                }
            }
        }
        Py_DECREF(hashlib);
    }
    if (hex.empty()) PyErr_Clear();
    return hex;
}

// The thumbnail file name (no directory) for a part id: <stem>.png. The caller must hold
// the GIL. Returns empty only if the SHA-256 fallback failed (extremely unlikely).
std::string ThumbnailNameFor(const std::string& partId) {
    std::string stem = ThumbnailStem(partId);
    if (stem.empty()) return std::string();
    return stem + ".png";
}

// Immediate parent-folder name of a path-like id (empty if there is none). Mirrors the
// subfolder that hoops_ai's generate_images creates under images_out_dir.
std::string PathParentName(const std::string& id) {
    fs::path parent = U8Path(id).parent_path();
    if (parent.empty()) return std::string();
    return parent.filename().u8string();
}

// Relative thumbnail path that hoops_ai's generate_images produces for a CAD file path:
//   <parent folder name>/<file stem>_white.png   (forward slashes; parent omitted if none).
// This reproduces the naming observed from embed_shape / embed_shape_batch with
// specifications={"generate_images": True, ...}; the authoritative value is otherwise read
// from EmbeddingBatch.metadata["png_paths"] at add time and stored in the record metadata.
std::string GenImagesRelThumb(const std::string& cadPath) {
    const std::string stem = U8Path(cadPath).stem().u8string();
    if (stem.empty()) return std::string();
    const std::string parent = PathParentName(cadPath);
    std::string rel = stem + "_white.png";
    if (!parent.empty()) rel = parent + "/" + rel;
    return rel;
}

// Build specifications={"generate_images": True, "images_out_dir": <dir>} (new reference) so
// the embedder renders each part's thumbnail in the same CAD load used for embedding. When
// timeLimitSeconds > 0 it also raises the per-item time budget so heavy files get more time
// before being dropped with a Timeout; <= 0 leaves the hoops_ai default (120 s).
//
// It sets ALL FOUR keys -- time_limit_overall AND time_limit_small/medium/large -- to the same
// value, matching the known-good configuration in the benchmark harness (bench_heavy.py,
// bench_step3_indexing.py). This was verified empirically: the per-item "Timeout (CUMULATIVE):
// item exceeded time_limit_s=120.0s" that killed heavy assemblies is governed by
// time_limit_overall; setting only the size buckets left it at the 120 s default, so a "900 s"
// pass-2 was still killed at 120 s. Setting time_limit_overall to a large per-item value does
// NOT abort the whole batch (bench: 16 files, time_limit_overall=1200 recovered every
// time-limited file; only genuine CAD errors failed). Returns nullptr on allocation failure.
// The caller must hold the GIL and owns the returned reference.
PyObject* BuildImageSpecs(const std::string& imagesDir, int timeLimitSeconds) {
    PyObject* specs = PyDict_New();
    if (!specs) return nullptr;
    PyDict_SetItemString(specs, "generate_images", Py_True);
    PyObject* d = PyUnicode_FromString(imagesDir.c_str());
    if (d) {
        PyDict_SetItemString(specs, "images_out_dir", d);
        Py_DECREF(d);
    }
    if (timeLimitSeconds > 0) {
        PyObject* t = PyFloat_FromDouble(static_cast<double>(timeLimitSeconds));
        if (t) {
            // Set all four keys to one budget. time_limit_overall governs the per-item CUMULATIVE
            // timeout (verified by benchmark); the size buckets are set too for forward-compat.
            PyDict_SetItemString(specs, "time_limit_overall", t);
            PyDict_SetItemString(specs, "time_limit_small", t);
            PyDict_SetItemString(specs, "time_limit_medium", t);
            PyDict_SetItemString(specs, "time_limit_large", t);
            Py_DECREF(t);
        }
    }
    return specs;
}

// Forward declaration; the definition lives later in this file (opt-in diagnostic logging).
void WriteDiagLog(const std::string& line);

// Bridge-side automatic worker-count selection for a batch embed (used when the caller passes
// numWorkers <= 0). hoops_ai's own auto-detect (num_workers=None) uses nearly all logical CPUs,
// but each spawned worker reloads its own copy of the ~2 GB embedding model, so on RAM-bound
// machines all-core parallelism oversubscribes memory: past ~8 workers throughput DROPS and
// per-item 120 s timeouts start dropping bodies from the index (data loss). Benchmarks (100
// mechcad files, 14C/20T / 32 GB) peaked at 8 workers (2.48x vs sequential) and regressed beyond
// that; all-core auto was 0.78x (slower than sequential). This returns a bounded worker count:
//   * 1 for small batches (spawn + per-worker model load is not amortized -- 20 files ran fastest
//     at 1 worker), otherwise
//   * min(maxWorkers, logicalCores/2, (availableRAM - reserve) / modelFootprint).
// All three limits are tunable at runtime (no rebuild) via environment variables:
//   HOOPS_AI_MAX_WORKERS         (default 8)
//   HOOPS_AI_MODEL_FOOTPRINT_MB  (default 2048, estimated per-worker model+runtime footprint)
//   HOOPS_AI_MIN_FILES_PARALLEL  (default 32, below which the batch runs on a single worker)
int ComputeAutoWorkers(int fileCount) {
    auto envInt = [](const char* name, int def) -> int {
        const char* v = std::getenv(name);
        if (!v || !*v) return def;
        char* end = nullptr;
        long n = std::strtol(v, &end, 10);
        if (end == v || n <= 0) return def;
        return static_cast<int>(n);
    };
    const int maxWorkers = envInt("HOOPS_AI_MAX_WORKERS", 8);
    const int modelMB = envInt("HOOPS_AI_MODEL_FOOTPRINT_MB", 2048);
    const int minFilesParallel = envInt("HOOPS_AI_MIN_FILES_PARALLEL", 32);

    if (fileCount < minFilesParallel) return 1;

    // Core-based cap: half the logical CPUs approximates the physical/performance cores and
    // avoids E-core / SMT oversubscription (hardware_concurrency() returns 0 when unknown).
    unsigned hw = std::thread::hardware_concurrency();
    int coreCap = (hw >= 2u) ? static_cast<int>(hw / 2u) : 1;

    // RAM-based cap: every worker keeps its own model copy, so bound by available physical RAM
    // minus a reserve for the host process and OS.
    int ramCap = maxWorkers;
#if defined(_WIN32)
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        const unsigned long long reserveBytes = 2ull * 1024 * 1024 * 1024; // 2 GB host/OS reserve
        unsigned long long avail = (ms.ullAvailPhys > reserveBytes) ? (ms.ullAvailPhys - reserveBytes) : 0ull;
        unsigned long long perWorker = static_cast<unsigned long long>(modelMB) * 1024ull * 1024ull;
        if (perWorker > 0ull) ramCap = static_cast<int>(avail / perWorker);
    }
#endif

    int n = maxWorkers;
    if (coreCap < n) n = coreCap;
    if (ramCap < n) n = ramCap;
    if (n < 1) n = 1;
    return n;
}

// Look up the relative thumbnail path stored in a record's metadata by scanning
// FaissVectorStore.iter_metadata() for the matching id (it exposes no get-by-id accessor).
// Returns empty when the id or the "thumbnail" key is absent. The caller must hold the GIL.
std::string LookupThumbFromMetadata(PyObject* vs, const std::string& id) {
    PyObject* metaIter = PyObject_CallMethod(vs, "iter_metadata", nullptr);
    if (!metaIter) { PyErr_Clear(); return std::string(); }
    PyObject* iter = PyObject_GetIter(metaIter);
    Py_DECREF(metaIter);
    if (!iter) { PyErr_Clear(); return std::string(); }

    std::string found;
    PyObject* item = nullptr;
    while ((item = PyIter_Next(iter)) != nullptr) {
        if (PyDict_Check(item)) {
            PyObject* fid = PyDict_GetItemString(item, "file_id"); // borrowed
            if (!fid) fid = PyDict_GetItemString(item, "id");      // borrowed
            if (fid && PyUnicode_Check(fid)) {
                const char* f = PyUnicode_AsUTF8(fid);
                if (f && id == f) {
                    PyObject* th = PyDict_GetItemString(item, "thumbnail"); // borrowed
                    if (th && PyUnicode_Check(th)) {
                        const char* t = PyUnicode_AsUTF8(th);
                        if (t) found = t;
                    }
                    Py_DECREF(item);
                    break;
                }
            }
        }
        Py_DECREF(item);
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) PyErr_Clear();
    return found;
}

// Atomically persist vs into <base>.faiss + <base>.meta by saving to a temp base (in the same
// parent directory) and renaming over the targets (mirrors _save_named_index_atomic in the
// Web API core.py). The caller must hold the GIL. Returns false with errOut set on failure.
bool SaveIndexAtomic(PyObject* vs, const std::string& base, std::string& errOut) {
    const fs::path parent = U8Path(base).parent_path();
    const std::string tmpBase =
        NormalizeSlashes((parent / ("_tmp_" + RandomHex(8))).u8string());

    PyObject* saveResult = PyObject_CallMethod(vs, "save", "(s)", tmpBase.c_str());
    if (!saveResult) {
        errOut = "save failed: " + FetchPythonError();
        return false;
    }
    Py_DECREF(saveResult);

    std::error_code ec;
    const char* exts[] = {".faiss", ".meta"};
    for (const char* ext : exts) {
        fs::path from = U8Path(tmpBase + ext);
        fs::path to   = U8Path(base + ext);
        if (!fs::exists(from, ec)) continue; // some stores may not emit .meta
        ec.clear();
        fs::rename(from, to, ec);
        if (ec) {
            // Cross-device or overwrite race: fall back to copy + remove.
            ec.clear();
            fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                errOut = "rename index file failed (" + std::string(ext) + "): " + SysErrUtf8(ec);
                fs::remove(from, ec);
                return false;
            }
            fs::remove(from, ec);
        }
    }
    // Clean up any leftover temp files (e.g. an unexpected extra extension).
    fs::remove(U8Path(tmpBase + ".faiss"), ec);
    fs::remove(U8Path(tmpBase + ".meta"), ec);
    return true;
}

// Shared implementation for HoopsAI_AddCADToIndex (one path) and HoopsAI_AddCADFolderToIndex
// (N paths). Embeds the input files with embedder.embed_shape_batch(paths, ...,
// specifications={"generate_images": True, "images_out_dir": <index folder>}) so hoops_ai
// renders each part's thumbnail during the SAME CAD load used for embedding (this replaces the
// former separate GenerateThumbnailFile/exportStreamCache pass and its double CAD load). One
// VectorRecord is built per returned row (embed_shape_batch yields one row per body, so a
// multi-body file contributes several rows that share the file path as id, matching TechSoft's
// per-body assembly indexing); each row is L2-normalized as before. Re-registered ids use
// delete -> upsert to avoid FAISS duplicates. The index is saved atomically exactly once.
//
// The authoritative thumbnail path is read from EmbeddingBatch.metadata["png_paths"][i] (relative
// to images_out_dir, e.g. "<parent>/<stem>_white.png") and stored in the record metadata; if it
// is unavailable the deterministic GenImagesRelThumb() rule is used as a fallback.
//
// idOverride (optional; may be empty) maps an input path to the record id to register it under
// (single-add partId support); when a path is absent the path itself is the id. numWorkers <= 0
// means "let the bridge choose" -- ComputeAutoWorkers() picks a bounded count (1 for small
// batches, else min(cap, cores/2, RAM/model)) instead of forwarding None to hoops_ai, whose
// all-core auto-detect oversubscribes RAM and can run slower than sequential; an explicit
// numWorkers > 0 is honored as-is. Either way this is safe because HoopsAI_Initialize repoints
// multiprocessing at a real python.exe. If that configuration failed (g_mpConfigured == false)
// numWorkers is clamped to 1 here. Outputs:
//   outAdded  = number of distinct input files embedded, outFailed = fileCount - outAdded.
//   outFailedPathsJoined (optional) = newline-joined inputs not echoed back in batch.ids, only
//     when its size matches outFailed. outIndexCount = total entries after the batch.
//   firstWarning carries the first non-fatal note (currently unused; reserved).
// The caller must hold the GIL. Returns false with err set on a hard failure.

// ---- Live progress forwarding (feature B) ----------------------------------------------------
//
// hoops_ai renders a tqdm bar on stderr while embed_shape_batch runs. To surface that to a native
// caller we temporarily replace sys.stderr with a small Python shim (kProgressShimPy) that mirrors
// every write to the real stderr AND parses the tqdm line, calling back into Bridge_ReportProgress
// which invokes the registered C callback. The shim reports isatty()==True so tqdm emits dynamic
// (incremental) updates even when the real stderr is a non-tty (NUL in a GUI host).

// C function exposed to the Python shim as report(phase, done, total, errors, heavy).
PyObject* Bridge_ReportProgress(PyObject* /*self*/, PyObject* args) {
    int phase = -1, done = 0, total = 0, errors = -1, heavy = -1;
    if (!PyArg_ParseTuple(args, "iiiii", &phase, &done, &total, &errors, &heavy))
        return nullptr;
    // Called with the GIL held on the worker thread. The registered callback is expected to be
    // cheap and thread-safe (e.g. post a queued Qt signal), so we invoke it directly.
    if (g_progressCb)
        g_progressCb(phase, done, total, errors, heavy, g_progressUser);
    Py_RETURN_NONE;
}
PyMethodDef kProgressMethodDef = {
    "report", Bridge_ReportProgress, METH_VARARGS,
    "Bridge internal: forward parsed tqdm progress to the native callback."
};

const char* kProgressShimPy = R"PY(
import re
class _BridgeProgressStderr:
    _re_frac  = re.compile(r'(\d+)\s*/\s*(\d+)')
    _re_err   = re.compile(r'errors=(\d+)')
    _re_heavy = re.compile(r'heavy=(\d+)')
    def __init__(self, real, report):
        self._real = real
        self._report = report
        self._last = None
    def write(self, s):
        try:
            if self._real is not None:
                self._real.write(s)
        except Exception:
            pass
        try:
            if s and (('Computing embeddings' in s) or ('Heavy Files' in s)):
                phase = 1 if ('Heavy Files' in s) else 0
                m = self._re_frac.search(s)
                if m:
                    done = int(m.group(1)); total = int(m.group(2))
                    me = self._re_err.search(s);   errors = int(me.group(1)) if me else -1
                    mh = self._re_heavy.search(s); heavy  = int(mh.group(1)) if mh else -1
                    key = (phase, done, total, errors, heavy)
                    if key != self._last:
                        self._last = key
                        self._report(phase, done, total, errors, heavy)
        except Exception:
            pass
        return len(s) if s else 0
    def flush(self):
        try:
            if self._real is not None:
                self._real.flush()
        except Exception:
            pass
    def isatty(self):
        return True
    def fileno(self):
        if self._real is not None:
            return self._real.fileno()
        raise OSError('no fileno')
    def __getattr__(self, name):
        return getattr(self._real, name)
)PY";

// Lazily build and cache the shim class object. Returns a borrowed reference (owned by
// g_progressShimClass) or nullptr on failure. Caller holds the GIL.
PyObject* EnsureProgressShimClass() {
    if (g_progressShimClass)
        return g_progressShimClass;
    PyObject* globals = PyDict_New();
    if (!globals)
        return nullptr;
    PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());
    PyObject* r = PyRun_String(kProgressShimPy, Py_file_input, globals, globals);
    if (!r) {
        PyErr_Clear();
        Py_DECREF(globals);
        return nullptr;
    }
    Py_DECREF(r);
    PyObject* cls = PyDict_GetItemString(globals, "_BridgeProgressStderr"); // borrowed
    if (cls) {
        Py_INCREF(cls);
        g_progressShimClass = cls;
    }
    Py_DECREF(globals);
    return g_progressShimClass;
}

// RAII-style installer: swaps sys.stderr for the shim on construction (when a callback is set)
// and restores it on destruction, even if the embed call throws. Caller holds the GIL.
struct ProgressStderrGuard {
    PyObject* oldStderr = nullptr;
    PyObject* shim = nullptr;
    bool installed = false;

    ProgressStderrGuard() {
        if (!g_progressCb)
            return;
        PyObject* cls = EnsureProgressShimClass();
        if (!cls)
            return;
        oldStderr = PySys_GetObject("stderr"); // borrowed, may be nullptr/None
        Py_XINCREF(oldStderr);
        PyObject* report = PyCFunction_New(&kProgressMethodDef, nullptr);
        if (!report) {
            PyErr_Clear();
            Py_XDECREF(oldStderr);
            oldStderr = nullptr;
            return;
        }
        PyObject* realArg = oldStderr ? oldStderr : Py_None;
        shim = PyObject_CallFunctionObjArgs(cls, realArg, report, nullptr);
        Py_DECREF(report);
        if (!shim) {
            PyErr_Clear();
            Py_XDECREF(oldStderr);
            oldStderr = nullptr;
            return;
        }
        PySys_SetObject("stderr", shim); // sys takes its own ref
        installed = true;
    }
    ~ProgressStderrGuard() {
        if (!installed)
            return;
        PySys_SetObject("stderr", oldStderr ? oldStderr : Py_None); // sys drops the shim ref
        Py_XDECREF(oldStderr);
        Py_XDECREF(shim);
    }
    bool active() const { return installed; }
};

bool AddPathsBatchCore(PyObject* vs,
                       const std::vector<std::string>& inputPaths,
                       const std::unordered_map<std::string, std::string>& idOverride,
                       int numWorkers,
                       int timeLimitSeconds,
                       int* outAdded, int* outFailed,
                       std::string* outFailedPathsJoined,
                       int* outIndexCount,
                       std::string& firstWarning,
                       std::string& err) {
    (void)firstWarning;
    if (outAdded) *outAdded = 0;
    if (outFailed) *outFailed = 0;
    if (outIndexCount) *outIndexCount = 0;
    if (outFailedPathsJoined) outFailedPathsJoined->clear();

    // Safety clamp: if HoopsAI_Initialize could not repoint multiprocessing at a real python.exe
    // (see ConfigureMultiprocessingSpawn), then more than one worker -- or the None/auto-detect
    // default requested via numWorkers <= 0 -- would spawn processes off the HOST executable and
    // relaunch it. Fall back to a single sequential worker and note it as a non-fatal warning.
    if (!g_mpConfigured && numWorkers != 1) {
        firstWarning = "parallel workers disabled: multiprocessing.set_executable was not "
                       "configured, so embedding runs with a single worker";
        numWorkers = 1;
    }

    const int fileCount = static_cast<int>(inputPaths.size());

    // Auto worker selection: numWorkers <= 0 means "let the bridge decide". Resolve it to a
    // bounded count here (see ComputeAutoWorkers) rather than forwarding None to hoops_ai, whose
    // all-logical-core auto-detect oversubscribes RAM (each worker reloads the ~2 GB model) and,
    // past ~8 workers, both slows the batch and triggers per-item timeouts that drop bodies. This
    // only runs when multiprocessing is safe (g_mpConfigured); the clamp above already forced 1
    // otherwise. An explicit numWorkers > 0 from the caller is always honored as-is.
    if (numWorkers <= 0) {
        numWorkers = ComputeAutoWorkers(fileCount);
        WriteDiagLog("[AddFolder] auto numWorkers=" + std::to_string(numWorkers)
                     + " (fileCount=" + std::to_string(fileCount) + ")");
    }

    // Thumbnails are rendered by hoops_ai into the per-index folder; ensure it exists first.
    const std::string imgDir = ThumbnailsDirOf(g_indexBase);
    if (!EnsureDir(U8Path(imgDir), err)) return false;

    // Build the Python list of input paths.
    PyObject* pyPaths = PyList_New(fileCount);
    if (!pyPaths) { err = "PyList_New failed"; return false; }
    for (int i = 0; i < fileCount; ++i) {
        PyObject* s = PyUnicode_FromString(inputPaths[static_cast<size_t>(i)].c_str());
        if (!s) { Py_DECREF(pyPaths); err = "build path list failed: " + FetchPythonError(); return false; }
        PyList_SET_ITEM(pyPaths, i, s); // Transfer the reference
    }

    // batch = embedder.embed_shape_batch(pyPaths, show_progress=False, specifications=specs
    //                                    [, num_workers=N])
    PyObject* method = PyObject_GetAttrString(g_embedder, "embed_shape_batch");
    if (!method) {
        Py_DECREF(pyPaths);
        err = "resolve embed_shape_batch failed: " + FetchPythonError();
        return false;
    }
    PyObject* specs = BuildImageSpecs(imgDir, timeLimitSeconds);
    if (!specs) {
        Py_DECREF(method);
        Py_DECREF(pyPaths);
        err = "build specifications failed";
        return false;
    }
    PyObject* args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, pyPaths); // Transfer the reference (pyPaths owned by args now)
    PyObject* kwargs = PyDict_New();
    // Install the tqdm-capturing stderr shim and enable the progress bar only when a native
    // progress callback is registered; otherwise keep the bar suppressed (show_progress=False).
    ProgressStderrGuard progressGuard;
    PyDict_SetItemString(kwargs, "show_progress", progressGuard.active() ? Py_True : Py_False);
    PyDict_SetItemString(kwargs, "specifications", specs);
    if (numWorkers > 0) {
        PyObject* nw = PyLong_FromLong(numWorkers);
        PyDict_SetItemString(kwargs, "num_workers", nw);
        Py_DECREF(nw);
    }
    PyObject* batch = PyObject_Call(method, args, kwargs);
    Py_DECREF(method);
    Py_DECREF(args);   // releases pyPaths
    Py_DECREF(kwargs);
    Py_DECREF(specs);
    if (!batch) { err = "embed_shape_batch failed: " + FetchPythonError(); return false; }

    // ids = batch.ids ; valuesList = batch.values.tolist() ; png_paths = batch.metadata["png_paths"]
    PyObject* idsObj = PyObject_GetAttrString(batch, "ids");
    PyObject* valuesAttr = PyObject_GetAttrString(batch, "values");
    if (!idsObj || !valuesAttr) {
        Py_XDECREF(idsObj);
        Py_XDECREF(valuesAttr);
        Py_DECREF(batch);
        err = "read batch ids/values failed: " + FetchPythonError();
        return false;
    }
    PyObject* valuesList = nullptr;
    if (PyObject_HasAttrString(valuesAttr, "tolist")) {
        valuesList = PyObject_CallMethod(valuesAttr, "tolist", nullptr);
    } else {
        Py_INCREF(valuesAttr);
        valuesList = valuesAttr;
    }
    Py_DECREF(valuesAttr);
    if (!valuesList) {
        Py_DECREF(idsObj);
        Py_DECREF(batch);
        err = "batch.values.tolist failed: " + FetchPythonError();
        return false;
    }

    // png_paths is optional; a missing/short list just falls back to the deterministic rule.
    // kind (per row: "part" for single-body files, "assembly" for multi-body) is likewise optional;
    // it is stored in each record's metadata so part search can exclude assemblies.
    PyObject* pngPathsSeq = nullptr;
    PyObject* kindSeq = nullptr;
    PyObject* metaObj = PyObject_GetAttrString(batch, "metadata");
    if (metaObj) {
        if (PyDict_Check(metaObj)) {
            PyObject* pp = PyDict_GetItemString(metaObj, "png_paths"); // borrowed
            if (pp) pngPathsSeq = PySequence_Fast(pp, "png_paths is not a sequence");
            PyObject* kd = PyDict_GetItemString(metaObj, "kind"); // borrowed
            if (kd) kindSeq = PySequence_Fast(kd, "kind is not a sequence");
        }
        Py_DECREF(metaObj);
    }
    if (!pngPathsSeq) PyErr_Clear();
    if (!kindSeq) PyErr_Clear();
    const Py_ssize_t nKind = kindSeq ? PySequence_Fast_GET_SIZE(kindSeq) : 0;

    PyObject* idsSeq = PySequence_Fast(idsObj, "batch.ids is not a sequence");
    PyObject* valSeq = PySequence_Fast(valuesList, "batch.values is not a sequence");
    Py_DECREF(idsObj);
    Py_DECREF(valuesList);
    if (!idsSeq || !valSeq) {
        Py_XDECREF(idsSeq);
        Py_XDECREF(valSeq);
        Py_XDECREF(pngPathsSeq);
        Py_XDECREF(kindSeq);
        Py_DECREF(batch);
        err = "batch ids/values not iterable: " + FetchPythonError();
        return false;
    }

    const Py_ssize_t nIds = PySequence_Fast_GET_SIZE(idsSeq);
    const Py_ssize_t nVals = PySequence_Fast_GET_SIZE(valSeq);
    const Py_ssize_t nPng = pngPathsSeq ? PySequence_Fast_GET_SIZE(pngPathsSeq) : 0;
    const Py_ssize_t n = (nIds < nVals) ? nIds : nVals;

    // Pre-fetch the existing ids once so re-registration uses delete -> upsert.
    PyObject* existingIds = PyObject_CallMethod(vs, "get_ids", nullptr);
    if (!existingIds) {
        Py_DECREF(idsSeq);
        Py_DECREF(valSeq);
        Py_XDECREF(pngPathsSeq);
        Py_XDECREF(kindSeq);
        Py_DECREF(batch);
        err = "get_ids failed: " + FetchPythonError();
        return false;
    }

    PyObject* recList = PyList_New(0);
    PyObject* delList = PyList_New(0);
    std::unordered_set<std::string> addedInputPaths; // distinct input files actually embedded
    const std::string registeredAt = UtcTimestamp();

    bool rowOk = true;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* idItem = PySequence_Fast_GET_ITEM(idsSeq, i); // borrowed
        const char* idc = PyUnicode_AsUTF8(idItem);
        if (!idc || !*idc) { PyErr_Clear(); continue; }
        const std::string inputPath = idc; // embed_shape_batch echoes back the input path

        PyObject* rowItem = PySequence_Fast_GET_ITEM(valSeq, i); // borrowed (list of floats)
        std::vector<float> vec;
        std::string vErr;
        if (!PyListToFloatVector(rowItem, vec, vErr)) {
            rowOk = false;
            err = "parse batch row failed: " + vErr;
            break;
        }
        L2Normalize(vec);

        // Record id: the caller-provided override for this input path, else the path itself.
        std::string recordId = inputPath;
        auto ov = idOverride.find(inputPath);
        if (ov != idOverride.end() && !ov->second.empty()) recordId = ov->second;

        std::string filename = inputPath;
        size_t slashPos = filename.find_last_of("/\\");
        if (slashPos != std::string::npos) filename = filename.substr(slashPos + 1);

        // Authoritative thumbnail path from png_paths[i]; else the deterministic rule.
        std::string thumbRel;
        if (pngPathsSeq && i < nPng) {
            PyObject* pth = PySequence_Fast_GET_ITEM(pngPathsSeq, i); // borrowed (Path or str)
            PyObject* s = PyObject_Str(pth);
            if (s) {
                const char* u = PyUnicode_AsUTF8(s);
                if (u) thumbRel = NormalizeSlashes(u);
                Py_DECREF(s);
            } else {
                PyErr_Clear();
            }
        }
        if (thumbRel.empty()) thumbRel = GenImagesRelThumb(inputPath);

        // kind[i] ("part"/"assembly") aligned with batch.ids; empty when unavailable.
        std::string kind;
        if (kindSeq && i < nKind) {
            PyObject* kd = PySequence_Fast_GET_ITEM(kindSeq, i); // borrowed
            if (kd && PyUnicode_Check(kd)) {
                const char* ku = PyUnicode_AsUTF8(kd);
                if (ku) kind = ku;
            }
        }

        std::string recErr;
        PyObject* recObj = BuildVectorRecord(recordId, filename, thumbRel, registeredAt, kind, vec, recErr);
        if (!recObj) {
            rowOk = false;
            err = recErr;
            break;
        }
        PyList_Append(recList, recObj);
        Py_DECREF(recObj);

        // Schedule a delete when the record id is already present (re-registration).
        PyObject* pyId = PyUnicode_FromString(recordId.c_str());
        const int contains = PySequence_Contains(existingIds, pyId);
        if (contains == 1) PyList_Append(delList, pyId);
        Py_DECREF(pyId);

        addedInputPaths.insert(inputPath);
    }
    Py_DECREF(existingIds);
    Py_DECREF(idsSeq);
    Py_DECREF(valSeq);
    Py_XDECREF(pngPathsSeq);
    Py_XDECREF(kindSeq);
    Py_DECREF(batch);

    if (!rowOk) {
        Py_DECREF(recList);
        Py_DECREF(delList);
        return false;
    }

    // Delete duplicates (if any), then upsert everything in one call.
    if (PyList_Size(delList) > 0) {
        PyObject* delRes = PyObject_CallMethod(vs, "delete", "(O)", delList);
        if (!delRes) {
            Py_DECREF(recList);
            Py_DECREF(delList);
            err = "delete (re-registration) failed: " + FetchPythonError();
            return false;
        }
        Py_DECREF(delRes);
    }
    Py_DECREF(delList);

    if (PyList_Size(recList) > 0) {
        PyObject* upRes = PyObject_CallMethod(vs, "upsert", "(O)", recList);
        if (!upRes) {
            Py_DECREF(recList);
            err = "upsert failed: " + FetchPythonError();
            return false;
        }
        Py_DECREF(upRes);
    }
    Py_DECREF(recList);

    // Persist atomically ONCE for the whole batch.
    if (!SaveIndexAtomic(vs, g_indexBase, err)) return false;

    PyObject* newIdsObj = PyObject_CallMethod(vs, "get_ids", nullptr);
    if (newIdsObj) {
        if (outIndexCount) *outIndexCount = static_cast<int>(PySequence_Size(newIdsObj));
        Py_DECREF(newIdsObj);
    }

    const int added = static_cast<int>(addedInputPaths.size());
    const int failed = fileCount - added;
    if (outAdded) *outAdded = added;
    if (outFailed) *outFailed = failed;

    // Best-effort list of failed input paths: inputs not echoed back in batch.ids. Only emit it
    // when the diff size matches the failed count, so a path-normalization mismatch never yields
    // a misleading list (the counts above are always correct).
    if (outFailedPathsJoined && failed > 0) {
        std::string joined;
        int diff = 0;
        for (const std::string& p : inputPaths) {
            if (addedInputPaths.find(p) == addedInputPaths.end()) {
                ++diff;
                if (!joined.empty()) joined += "\n";
                joined += p;
            }
        }
        if (diff == failed) *outFailedPathsJoined = joined;
    }

    return true;
}

// Append a diagnostic line to %TEMP%\hoops_ai_bridge_diag.log (or ./hoops_ai_bridge_diag.log if
// TEMP is unset). OPT-IN: does nothing unless the environment variable HOOPS_AI_BRIDGE_DIAG is
// set, so production runs write no files. Best-effort: any failure is ignored. Used to debug the
// multiprocessing/spawn and stdio setup and the folder-add outcome from GUI hosts with no console.
void WriteDiagLog(const std::string& line) {
    try {
#if defined(_WIN32)
        if (GetEnvironmentVariableA("HOOPS_AI_BRIDGE_DIAG", nullptr, 0) == 0) return; // not set
#else
        if (!std::getenv("HOOPS_AI_BRIDGE_DIAG")) return;
#endif
        std::string dir;
#if defined(_WIN32)
        char buf[MAX_PATH] = {0};
        DWORD n = GetEnvironmentVariableA("TEMP", buf, sizeof(buf));
        if (n > 0 && n < sizeof(buf)) dir = buf;
#else
        const char* t = std::getenv("TMPDIR");
        if (t) dir = t; else dir = "/tmp";
#endif
        fs::path p = dir.empty() ? fs::path("hoops_ai_bridge_diag.log")
                                 : (fs::path(dir) / "hoops_ai_bridge_diag.log");
        std::ofstream f(p, std::ios::app);
        if (f) f << line << "\n";
    } catch (...) {
        // best-effort logging only
    }
}

// Make multiprocessing's 'spawn' start method (used by hoops_ai's embed_shape_batch parallel
// executor, which fixes start_method='spawn') safe inside this EMBEDDED interpreter. By default
// multiprocessing launches each worker with sys.executable, which here is the HOST executable
// (the application that loaded this DLL) -- so every worker would relaunch the whole host.
// We repoint multiprocessing at the real python.exe of the active interpreter (found next to
// sys.base_exec_prefix / sys.exec_prefix) and neutralize sys.argv[0] so spawned children do not
// try to re-import/re-run a "__main__" host script. On success num_workers > 1 (and the
// None/auto-detect default, i.e. numWorkers <= 0) can be used safely; on failure the batch add
// code clamps num_workers to 1. The caller MUST hold the GIL. `diag` receives a human-readable
// summary of the resolved values for logging.
bool ConfigureMultiprocessingSpawn(std::string& err, std::string& diag) {
    static const char* kCode =
        "import sys, os, multiprocessing\n"
        "import multiprocessing.spawn as _sp\n"
        "_ok = False\n"
        "_exe = ''\n"
        "_names = ('python.exe',) if os.name == 'nt' else ('python3', 'python')\n"
        "for _base in (getattr(sys, 'base_exec_prefix', ''), getattr(sys, 'exec_prefix', '')):\n"
        "    if not _base:\n"
        "        continue\n"
        "    for _n in _names:\n"
        "        _cand = os.path.join(_base, _n) if os.name == 'nt' else os.path.join(_base, 'bin', _n)\n"
        "        if os.path.isfile(_cand):\n"
        "            _exe = _cand\n"
        "            break\n"
        "    if _exe:\n"
        "        break\n"
        "if _exe:\n"
        "    multiprocessing.set_executable(_exe)\n"
        "    # A spawned child inspects sys.argv[0] to decide whether to re-import a __main__\n"
        "    # script; '' disables that (there is no host script to re-run).\n"
        "    if getattr(sys, 'argv', None):\n"
        "        sys.argv[0] = ''\n"
        "    else:\n"
        "        sys.argv = ['']\n"
        "    _ok = True\n"
        "try:\n"
        "    _spawn_exe = _sp.get_executable()\n"
        "except Exception as _e:\n"
        "    _spawn_exe = 'ERR:' + repr(_e)\n"
        "_diag = ('sys.executable=' + repr(getattr(sys, 'executable', None))\n"
        "         + ' base_exec_prefix=' + repr(getattr(sys, 'base_exec_prefix', None))\n"
        "         + ' exec_prefix=' + repr(getattr(sys, 'exec_prefix', None))\n"
        "         + ' chosen_exe=' + repr(_exe)\n"
        "         + ' spawn.get_executable=' + repr(_spawn_exe)\n"
        "         + ' argv=' + repr(getattr(sys, 'argv', None)))\n";
    PyObject* d = PyDict_New();
    if (!d) { err = "PyDict_New failed"; return false; }
    PyObject* r = PyRun_String(kCode, Py_file_input, d, d);
    if (!r) {
        err = "configure multiprocessing failed: " + FetchPythonError();
        Py_DECREF(d);
        return false;
    }
    Py_DECREF(r);
    PyObject* diagObj = PyDict_GetItemString(d, "_diag"); // borrowed
    if (diagObj) {
        const char* s = PyUnicode_AsUTF8(diagObj);
        if (s) diag = s;
    }
    PyObject* okObj = PyDict_GetItemString(d, "_ok"); // borrowed
    bool ok = (okObj && PyObject_IsTrue(okObj));
    if (!ok) err = "could not locate a real python interpreter next to "
                   "sys.base_exec_prefix / sys.exec_prefix";
    Py_DECREF(d);
    return ok;
}

#if defined(_WIN32)
// In a GUI (Windows subsystem) host there is no console, so the process standard handles are
// often invalid (NULL). CPython then initializes sys.stdout/sys.stderr/sys.stdin to None, and
// any code that writes to them crashes -- notably hoops_ai's parallel executor, which creates a
// tqdm progress bar (tqdm writes to sys.stderr) -> "AttributeError: 'NoneType' object has no
// attribute 'write'", which aborts embed_shape_batch(num_workers>1) immediately. This redirects
// each INVALID standard stream to the NUL device (both the CRT stream and the process std handle,
// so spawned worker python.exe children inherit a valid handle too). Console hosts keep their
// real streams untouched (only invalid handles are replaced). MUST run before Py_InitializeEx so
// the interpreter builds real sys.std* streams from the start. Returns a short diag string.
std::string EnsureValidStdStreamsWin() {
    struct Item { DWORD id; FILE* stream; const char* mode; const char* name; };
    const Item items[] = {
        { STD_INPUT_HANDLE,  stdin,  "r", "stdin"  },
        { STD_OUTPUT_HANDLE, stdout, "w", "stdout" },
        { STD_ERROR_HANDLE,  stderr, "w", "stderr" },
    };
    std::string diag = "stdfix:";
    for (const Item& it : items) {
        HANDLE h = GetStdHandle(it.id);
        if (h != NULL && h != INVALID_HANDLE_VALUE) {
            diag += std::string(" ") + it.name + "=ok";
            continue; // real console/redirected handle: leave it alone
        }
        FILE* f = nullptr;
        if (freopen_s(&f, "NUL", it.mode, it.stream) == 0 && f) {
            // Propagate to the process std handle so spawned children inherit a valid handle.
            intptr_t osfh = _get_osfhandle(_fileno(it.stream));
            if (osfh != -1) SetStdHandle(it.id, reinterpret_cast<HANDLE>(osfh));
            diag += std::string(" ") + it.name + "=NUL";
        } else {
            diag += std::string(" ") + it.name + "=FAIL";
        }
    }
    return diag;
}
#endif

// Python-level guard (all platforms): even when the OS std handles look valid (e.g. a GUI-
// subsystem exe launched from a console inherits the parent's handles), CPython embedded in a
// windowed host can still end up with sys.stdout/stderr/stdin == None because the CRT stdio fds
// are not wired up. hoops_ai's parallel executor writes a tqdm progress bar to sys.stderr, and
// tqdm crashes with "AttributeError: 'NoneType' object has no attribute 'write'" when it is None.
// This replaces any None standard stream with an os.devnull handle so the batch add never aborts.
// Must run after Py_InitializeEx (GIL held). Returns a short diag string. This is what actually
// covers the console-launched GUI case that EnsureValidStdStreamsWin (OS-handle based) misses.
std::string FixPyStdStreamsNone() {
    const char* names[] = { "stdin", "stdout", "stderr" };
    std::string diag = "pystdfix:";
    for (const char* n : names) {
        PyObject* o = PySys_GetObject(n); // borrowed, may be NULL
        bool isNone = (o == nullptr || o == Py_None);
        diag += std::string(" ") + n + "=" + (isNone ? "None" : "ok");
    }
    const char* code =
        "import sys, os\n"
        "for _n in ('stdin', 'stdout', 'stderr'):\n"
        "    if getattr(sys, _n, None) is None:\n"
        "        _f = open(os.devnull, 'r' if _n == 'stdin' else 'w')\n"
        "        setattr(sys, _n, _f)\n"
        "        setattr(sys, '__' + _n + '__', _f)\n";
    if (PyRun_SimpleString(code) != 0) {
        PyErr_Clear();
        diag += " apply=FAIL";
    } else {
        diag += " apply=ok";
    }
    return diag;
}

} // namespace

extern "C" HOOPSAI_API bool HoopsAI_Initialize(const char* venvSitePackages,
                                                const char* pythonHome,
                                                const char* licenseKey,
                                                char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (g_initialized) return true;

    try {
#if defined(_WIN32)
        // Resolve the base Python 3.12 home: explicit arg > HOOPS_AI_PYTHON_HOME
        // env var > PEP 514 registry (HKCU then HKLM). Only set it if a
        // candidate directory actually contains python312.dll.
        // Py_SetPythonHome keeps the pointer and reads it during Py_InitializeEx,
        // so wHome is kept alive (static) until after initialization completes.
        std::wstring resolvedHome = ResolvePythonHomeWin(pythonHome);
        bool pythonHomeResolved = !resolvedHome.empty();
        if (pythonHomeResolved) {
            static std::wstring wHome;
            wHome = resolvedHome;
            Py_SetPythonHome(wHome.c_str());
        }
#else
        // On Linux, the system/venv Python is normally used as-is, so explicitly setting PYTHONHOME is unnecessary.
        // If needed, the caller is expected to set the PYTHONHOME environment variable.
        (void)pythonHome;
#endif

#if defined(_WIN32)
        // Redirect any invalid standard stream (GUI host with no console) to NUL BEFORE the
        // interpreter starts, so sys.stdout/stderr/stdin are real streams and hoops_ai's parallel
        // executor (which writes a tqdm progress bar to sys.stderr) does not crash.
        std::string stdFixDiag = EnsureValidStdStreamsWin();
        WriteDiagLog("[Initialize] " + stdFixDiag);
#endif

        Py_InitializeEx(0); // Do not register signal handlers (intended for embedded use)
        if (!Py_IsInitialized()) {
            SetError(outErrorMsg, errorMsgSize, "Py_InitializeEx failed");
            return false;
        }

        // Guard against sys.stdout/stderr/stdin == None (windowed host). See FixPyStdStreamsNone.
        // This covers the case the OS-handle check above cannot (valid handles, but None Python
        // streams), which otherwise crashes hoops_ai's tqdm progress bar and aborts the batch add.
        WriteDiagLog("[Initialize] " + FixPyStdStreamsNone());
        // Insert the caller-provided venv site-packages at the front of sys.path
        // so hoops_ai can be imported. The bridge does not validate or auto-detect
        // it; resolving the location (HOOPS_AI_HOME in development, <exe>\..\.venv
        // in a redistribution layout) is the caller's responsibility.
        bool sitePackagesProvided = (venvSitePackages && *venvSitePackages);
        if (sitePackagesProvided) {
            PyObject* sysPath = PySys_GetObject("path"); // borrowed
            PyObject* pyPath = PyUnicode_FromString(venvSitePackages);
            PyList_Insert(sysPath, 0, pyPath);
            Py_DECREF(pyPath);
        }

        // Make multiprocessing 'spawn' safe before any embed_shape_batch(num_workers>1) runs:
        // repoint it at the real python.exe so parallel workers do not relaunch the host. Best
        // effort -- on failure g_mpConfigured stays false and the batch add clamps to 1 worker.
        {
            std::string mpErr, mpDiag;
            g_mpConfigured = ConfigureMultiprocessingSpawn(mpErr, mpDiag);
            // Not fatal: a false result simply disables parallel workers (single-worker adds
            // still work); the reason is left in mpErr for local debugging only.
            WriteDiagLog("[Initialize] g_mpConfigured=" + std::string(g_mpConfigured ? "1" : "0")
                         + " " + mpDiag + (mpErr.empty() ? "" : (" mpErr=" + mpErr)));
            (void)mpErr;
        }

        PyObject* hoopsAiModule = PyImport_ImportModule("hoops_ai");
        if (!hoopsAiModule) {
            std::string msg = "import hoops_ai failed: " + FetchPythonError();
            msg += "\n[hint] Pass the path (UTF-8) of a site-packages directory "
                   "that contains hoops_ai via venvSitePackages. The caller "
                   "resolves the location: from HOOPS_AI_HOME in development, or "
                   "from <exe>\\..\\.venv in a redistribution layout.";
#if defined(_WIN32)
            if (!pythonHomeResolved) {
                msg += "\n[hint] Could not locate the Python 3.12 installation "
                       "(python312.dll). Install Python 3.12 (python.org), or "
                       "specify its directory explicitly via the "
                       "HOOPS_AI_PYTHON_HOME environment variable.";
            }
#endif
            SetError(outErrorMsg, errorMsgSize, msg);
            return false;
        }

        // License setup: hoops_ai.set_license(key, validate=True)
        std::string key = (licenseKey && *licenseKey) ? licenseKey : "";
        if (!key.empty()) {
            PyObject* result = PyObject_CallMethod(hoopsAiModule, "set_license", "(sO)",
                                                    key.c_str(), Py_True);
            if (!result) {
                SetError(outErrorMsg, errorMsgSize, "set_license failed: " + FetchPythonError());
                Py_DECREF(hoopsAiModule);
                return false;
            }
            Py_DECREF(result);
        }
        Py_DECREF(hoopsAiModule);

        // Release the GIL now that start-up (import + license) is done, saving the main
        // thread state. Without this the GIL stays held by this thread and a
        // PyGILState_Ensure() from any other thread (e.g. a Qt worker rendering a thumbnail)
        // would deadlock. Every export re-acquires the GIL via PyGILState_Ensure/Release.
        g_mainThreadState = PyEval_SaveThread();

        g_initialized = true;
        return true;
    } catch (const std::exception& e) {
        SetError(outErrorMsg, errorMsgSize, std::string("C++ exception: ") + e.what());
        return false;
    }
}

extern "C" HOOPSAI_API bool HoopsAI_LoadMFRModel(const char* checkpointPath,
                                                  char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        // Already loaded with the same checkpoint -> reuse (avoid a costly rebuild).
        // A different (or first) path falls through to a real load below.
        const std::string reqPath = (checkpointPath && *checkpointPath) ? checkpointPath : "";
        if (g_mfrModel && reqPath == g_mfrCkptPath) { ok = true; break; }

        std::string err;
        PyObject* loader = GetSharedCADLoader(err); // CAD ACCESS: use the shared helper
        if (!loader) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }

        PyObject* mlMod = PyImport_ImportModule("hoops_ai.ml.EXPERIMENTAL");
        if (!mlMod) {
            SetError(outErrorMsg, errorMsgSize, "import hoops_ai.ml.EXPERIMENTAL failed: " + FetchPythonError());
            break;
        }

        PyObject* flowInferenceClass = PyObject_GetAttrString(mlMod, "FlowInference");
        PyObject* graphNodeClsClass = PyObject_GetAttrString(mlMod, "GraphNodeClassification");
        Py_DECREF(mlMod);
        if (!flowInferenceClass || !graphNodeClsClass) {
            SetError(outErrorMsg, errorMsgSize, "resolve classes failed: " + FetchPythonError());
            break;
        }

        // GraphNodeClassification(result_dir="."): output destination for MFR (logs, etc.)
        PyObject* kwargsFlowModel = Py_BuildValue("{s:s}", "result_dir", ".");
        PyObject* flowModel = PyObject_Call(graphNodeClsClass, PyTuple_New(0), kwargsFlowModel);
        Py_DECREF(kwargsFlowModel);
        Py_DECREF(graphNodeClsClass); // No longer needed, so release immediately

        if (!flowModel) {
            SetError(outErrorMsg, errorMsgSize, "construct flowmodel failed: " + FetchPythonError());
            Py_DECREF(flowInferenceClass);
            break;
        }

        // FlowInference(cad_loader=loader, flowmodel=flowModel) -- loader is shared (borrowed), so no DECREF is needed
        PyObject* kwargsInfer = Py_BuildValue("{s:O,s:O}", "cad_loader", loader, "flowmodel", flowModel);
        PyObject* inferenceModel = PyObject_Call(flowInferenceClass, PyTuple_New(0), kwargsInfer);
        Py_DECREF(kwargsInfer);
        Py_DECREF(flowModel);
        Py_DECREF(flowInferenceClass);
        if (!inferenceModel) {
            SetError(outErrorMsg, errorMsgSize, "construct FlowInference failed: " + FetchPythonError());
            break;
        }

        // model.load_from_checkpoint(checkpoint_path)
        PyObject* loadResult = PyObject_CallMethod(inferenceModel, "load_from_checkpoint", "(s)", checkpointPath);
        if (!loadResult) {
            SetError(outErrorMsg, errorMsgSize, "load_from_checkpoint failed: " + FetchPythonError());
            Py_DECREF(inferenceModel);
            break;
        }
        Py_DECREF(loadResult);

        // Swap in the new model only after it fully loaded, so a failed load
        // leaves the previously loaded model (if any) intact.
        Py_XDECREF(g_mfrModel);
        g_mfrModel = inferenceModel; // Keep for the lifetime of the process
        g_mfrCkptPath = reqPath;
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_RunMFRInference(const char* cadFilePath,
                                                      int* outLabels, int maxLabels,
                                                      int* outLabelCount,
                                                      char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (!g_mfrModel) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_LoadMFRModel was not called");
        return false;
    }
    if (outLabelCount) *outLabelCount = 0;

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        // ml_input = model.preprocess(cad_file_path)
        PyObject* mlInput = PyObject_CallMethod(g_mfrModel, "preprocess", "(s)", cadFilePath);
        if (!mlInput) {
            SetError(outErrorMsg, errorMsgSize, "preprocess failed: " + FetchPythonError());
            break;
        }

        // predictions, probabilities = model.predict_and_postprocess(ml_input)
        PyObject* predictResult = PyObject_CallMethod(g_mfrModel, "predict_and_postprocess", "(O)", mlInput);
        Py_DECREF(mlInput);
        if (!predictResult) {
            SetError(outErrorMsg, errorMsgSize, "predict_and_postprocess failed: " + FetchPythonError());
            break;
        }

        PyObject* predictions = PyTuple_Check(predictResult) ? PyTuple_GetItem(predictResult, 0) : predictResult;

        PyObject* asList = PyObject_CallMethod(predictions, "tolist", nullptr);
        Py_DECREF(predictResult);
        if (!asList) {
            SetError(outErrorMsg, errorMsgSize, "predictions.tolist() failed: " + FetchPythonError());
            break;
        }

        Py_ssize_t n = PyList_Size(asList);
        int count = 0;
        for (Py_ssize_t i = 0; i < n && count < maxLabels; ++i) {
            PyObject* item = PyList_GetItem(asList, i); // borrowed
            long labelVal = PyLong_Check(item) ? PyLong_AsLong(item)
                                                : static_cast<long>(PyFloat_AsDouble(item));
            outLabels[count++] = static_cast<int>(labelVal);
        }
        if (outLabelCount) *outLabelCount = count;

        Py_DECREF(asList);
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_LoadEmbeddingsModel(const char* checkpointPath,
                                                         char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        // Already loaded with the same checkpoint -> reuse (avoid a costly rebuild).
        // A different (or first) path falls through to a real (re)load below.
        const std::string reqPath = (checkpointPath && *checkpointPath) ? checkpointPath : "";
        if (g_embedder && reqPath == g_embedderCkptPath) { ok = true; break; }

        // register_model rejects a repeated model_name and there is no unregister
        // API, so reserve a fresh unique name for every (re)load attempt. The
        // counter advances even on failure so a retry never reuses a name that
        // register_model already consumed.
        const std::string modelName =
            "hoops_embeddings_user_" + std::to_string(++g_embModelSeq);

        std::string err;
        PyObject* loader = GetSharedCADLoader(err); // CAD ACCESS: use the shared helper
        if (!loader) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }

        PyObject* embMod = PyImport_ImportModule("hoops_ai.ml.embeddings");
        if (!embMod) {
            SetError(outErrorMsg, errorMsgSize, "import hoops_ai.ml.embeddings failed: " + FetchPythonError());
            break;
        }

        PyObject* embClass = PyObject_GetAttrString(embMod, "HOOPSEmbeddings");
        Py_DECREF(embMod);
        if (!embClass) {
            SetError(outErrorMsg, errorMsgSize, "resolve HOOPSEmbeddings failed: " + FetchPythonError());
            break;
        }

        // HOOPSEmbeddings.register_model(model_name=..., checkpoint_path=...)  (classmethod call)
        PyObject* registerMethod = PyObject_GetAttrString(embClass, "register_model");
        if (!registerMethod) {
            SetError(outErrorMsg, errorMsgSize, "resolve register_model failed: " + FetchPythonError());
            Py_DECREF(embClass);
            break;
        }
        PyObject* kwargsRegister = Py_BuildValue("{s:s,s:s}",
                                                  "model_name", modelName.c_str(),
                                                  "checkpoint_path", checkpointPath);
        PyObject* emptyArgs = PyTuple_New(0);
        PyObject* registerResult = PyObject_Call(registerMethod, emptyArgs, kwargsRegister);
        Py_DECREF(emptyArgs);
        Py_DECREF(kwargsRegister);
        Py_DECREF(registerMethod);
        if (!registerResult) {
            SetError(outErrorMsg, errorMsgSize, "register_model failed: " + FetchPythonError());
            Py_DECREF(embClass);
            break;
        }
        Py_DECREF(registerResult);

        // embedder = HOOPSEmbeddings(cad_loader=loader, model=<unique name>)
        PyObject* kwargsCtor = Py_BuildValue("{s:O,s:s}",
                                              "cad_loader", loader,
                                              "model", modelName.c_str());
        PyObject* embedder = PyObject_Call(embClass, PyTuple_New(0), kwargsCtor);
        Py_DECREF(kwargsCtor);
        Py_DECREF(embClass);
        if (!embedder) {
            SetError(outErrorMsg, errorMsgSize, "construct HOOPSEmbeddings failed: " + FetchPythonError());
            break;
        }

        // Swap in the new embedder only after it fully constructed, so a failed
        // load leaves the previously loaded embedder (if any) intact.
        Py_XDECREF(g_embedder);
        g_embedder = embedder; // Keep for the lifetime of the process
        g_embedderCkptPath = reqPath;
        g_currentEmbModelName = modelName;
        // Vectors from different models are not comparable, so close the current index
        // (in-memory release only; the index files on disk are NOT deleted, since they are
        // client-owned). After switching models, open a fresh index with HoopsAI_OpenIndex.
        CloseCurrentIndex();
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_ComputeEmbedding(const char* cadFilePath,
                                                      float* outVector, int maxLen, int* outDim,
                                                      char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (!g_embedder) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_LoadEmbeddingsModel was not called");
        return false;
    }
    if (outDim) *outDim = 0;

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        // The vector computation itself is factored out into ComputeEmbeddingVector() and is shared with
        // HoopsAI_CompareEmbeddings and the query side of HoopsAI_SearchIndex (same L2 normalization and
        // multi-body averaging logic). Index registration (HoopsAI_AddCADToIndex /
        // HoopsAI_AddCADFolderToIndex) does NOT use this; it stores one row per body via embed_shape_batch.
        std::string err;
        std::vector<float> finalVec;
        if (!ComputeEmbeddingVector(cadFilePath, finalVec, err)) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }

        int count = 0;
        for (size_t i = 0; i < finalVec.size() && static_cast<int>(count) < maxLen; ++i) {
            outVector[count++] = finalVec[i];
        }
        if (outDim) *outDim = count;
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_CompareEmbeddings(const char* cadFilePath1,
                                                       const char* cadFilePath2,
                                                       float* outSimilarity,
                                                       char* outErrorMsg, int errorMsgSize) {
    if (outSimilarity) *outSimilarity = 0.0f;

    constexpr int kMaxDim = 8192; // Upper bound with ample headroom for the HOOPS AI embedding dimension (currently 2048)
    std::vector<float> v1(kMaxDim), v2(kMaxDim);
    int dim1 = 0, dim2 = 0;

    // Same design as compare_embeddings in the Web API version: call compute_embedding twice, then apply L2 normalization + dot product.
    if (!HoopsAI_ComputeEmbedding(cadFilePath1, v1.data(), kMaxDim, &dim1, outErrorMsg, errorMsgSize)) {
        return false;
    }
    if (!HoopsAI_ComputeEmbedding(cadFilePath2, v2.data(), kMaxDim, &dim2, outErrorMsg, errorMsgSize)) {
        return false;
    }
    if (dim1 != dim2 || dim1 == 0) {
        SetError(outErrorMsg, errorMsgSize, "embedding dimension mismatch or empty");
        return false;
    }

    // L2 normalization (HoopsAI_ComputeEmbedding already returns normalized vectors, so this is
    // almost a no-op in practice, but normalize again just in case. The unused tail region remains
    // zero-filled, so it affects neither the norm calculation nor the dot product).
    L2Normalize(v1);
    L2Normalize(v2);

    // Cosine similarity = dot product of normalized vectors
    double dot = 0.0;
    for (int i = 0; i < dim1; ++i) dot += static_cast<double>(v1[i]) * v2[i];

    if (outSimilarity) *outSimilarity = static_cast<float>(dot);
    return true;
}

extern "C" HOOPSAI_API bool HoopsAI_AddCADToIndex(const char* cadFilePath,
                                                   const char* partId,
                                                   int* outIndexCount,
                                                   char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (!g_embedder) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_LoadEmbeddingsModel was not called");
        return false;
    }
    if (outIndexCount) *outIndexCount = 0;
    if (!cadFilePath || !*cadFilePath) {
        SetError(outErrorMsg, errorMsgSize, "cadFilePath is empty");
        return false;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        std::string err;

        // Operate on the current index; fail clearly if none is open (no implicit create).
        PyObject* vs = GetSharedIndex(err);
        if (!vs) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }

        // Add a single file through the shared batch core so its embedding AND thumbnail are
        // produced exactly like the folder add (one embed_shape_batch call with
        // generate_images). num_workers=1: a single file is embedded sequentially anyway
        // (hoops_ai runs batches < 4 files sequentially), so there is nothing to parallelize.
        const std::vector<std::string> paths = { cadFilePath };
        std::unordered_map<std::string, std::string> idOverride;
        if (partId && *partId) idOverride.emplace(cadFilePath, partId);

        int added = 0, failed = 0;
        std::string firstWarning;
        if (!AddPathsBatchCore(vs, paths, idOverride, /*numWorkers=*/1, /*timeLimitSeconds=*/0,
                               &added, &failed, /*outFailedPathsJoined=*/nullptr,
                               outIndexCount, firstWarning, err)) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }
        if (added == 0) {
            SetError(outErrorMsg, errorMsgSize,
                     "failed to embed CAD file: " + std::string(cadFilePath));
            break;
        }
        if (!firstWarning.empty())
            SetError(outErrorMsg, errorMsgSize, "warning: " + firstWarning);

        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

// Batch variant of HoopsAI_AddCADToIndex for "add a whole folder". Both delegate to
// AddPathsBatchCore: one embedder.embed_shape_batch(cad_path_list, ...,
// specifications={"generate_images": True, "images_out_dir": <index folder>}) call builds one
// VectorRecord per returned row, upserts them all in a single call, saves the index exactly
// ONCE, and lets hoops_ai render each thumbnail in the same CAD load. That removes the O(N^2)
// disk I/O of re-saving the full FAISS index per file and the former separate thumbnail pass.
//
// [Aggregation note] embed_shape_batch returns one row per body (batch.values row i for
// batch.ids[i]); each row is L2-normalized. Multi-body files therefore contribute several
// per-body rows that share the file path as id (matching TechSoft's per-body assembly
// indexing), rather than the single averaged vector used previously.
//
// numWorkers: parallel worker processes for embed_shape_batch. <= 0 lets the BRIDGE choose a
//   bounded worker count (ComputeAutoWorkers: 1 for small batches, else min(cap, cores/2,
//   availableRAM/model-footprint)); this is the recommended value. It deliberately does NOT
//   forward None to hoops_ai, whose all-logical-core auto-detect oversubscribes RAM -- each
//   worker reloads the ~2 GB model -- and past ~8 workers both slows the batch and drops bodies
//   via per-item timeouts (benchmarked peak was 8 workers, all-core was slower than sequential).
//   Pass an explicit N > 0 to override. The caps are tunable via HOOPS_AI_MAX_WORKERS /
//   HOOPS_AI_MODEL_FOOTPRINT_MB / HOOPS_AI_MIN_FILES_PARALLEL. HoopsAI_Initialize repoints
//   multiprocessing.set_executable() at a real python.exe so spawned workers launch a clean
//   interpreter instead of relaunching the host; if that setup failed the core clamps to 1.
// Each file is registered under its own path as the id (same default as
// HoopsAI_AddCADToIndex with partId == nullptr).
// outAddedCount / outFailedCount: distinct files successfully embedded+added / that failed.
// outFailedPaths (optional): newline-delimited UTF-8 list of the input paths that failed
//   (only filled when it can be derived unambiguously; the counts are always reported).
// outIndexCount: total entries in the index after the batch.
// Thumbnails are rendered by hoops_ai during embedding; a true return does not guarantee
// outErrorMsg is empty (it carries the first warning if any).
extern "C" HOOPSAI_API void HoopsAI_SetProgressCallback(HoopsAI_ProgressCallback callback,
                                                        void* userData) {
    // Plain-pointer store; serialize with the process-wide lock so it does not race a folder add
    // that is reading the globals under the same lock. No GIL needed (no Python objects touched).
    std::lock_guard<std::mutex> lock(g_pyMutex);
    g_progressCb = callback;
    g_progressUser = userData;
}

extern "C" HOOPSAI_API bool HoopsAI_AddCADFolderToIndex(const char* const* cadFilePaths,
                                                        int fileCount,
                                                        int numWorkers,
                                                        int timeLimitSeconds,
                                                        int* outAddedCount,
                                                        int* outFailedCount,
                                                        char* outFailedPaths, int outFailedPathsBufSize,
                                                        int* outIndexCount,
                                                        char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (outAddedCount) *outAddedCount = 0;
    if (outFailedCount) *outFailedCount = 0;
    if (outIndexCount) *outIndexCount = 0;
    if (outFailedPaths && outFailedPathsBufSize > 0) outFailedPaths[0] = '\0';

    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (!g_embedder) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_LoadEmbeddingsModel was not called");
        return false;
    }
    if (!cadFilePaths || fileCount <= 0) {
        SetError(outErrorMsg, errorMsgSize, "no CAD files given");
        return false;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;
    std::string firstWarning;

    do {
        std::string err;

        // Operate on the current index; fail clearly if none is open (no implicit create).
        PyObject* vs = GetSharedIndex(err);
        if (!vs) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }

        // Delegate to the shared core: one embed_shape_batch call embeds every file and lets
        // hoops_ai render each thumbnail (generate_images) in the same CAD load, then upserts
        // all rows and saves the index once. Folder adds register each file under its own path
        // (no id override).
        std::vector<std::string> inputPaths;
        inputPaths.reserve(static_cast<size_t>(fileCount));
        for (int i = 0; i < fileCount; ++i)
            inputPaths.emplace_back(cadFilePaths[i] ? cadFilePaths[i] : "");

        int added = 0, failed = 0;
        std::string failedJoined;
        WriteDiagLog("[AddFolder] enter fileCount=" + std::to_string(fileCount)
                     + " numWorkers=" + std::to_string(numWorkers)
                     + " timeLimitSeconds=" + std::to_string(timeLimitSeconds)
                     + " g_mpConfigured=" + std::string(g_mpConfigured ? "1" : "0"));
        if (!AddPathsBatchCore(vs, inputPaths, /*idOverride=*/{}, numWorkers, timeLimitSeconds,
                               &added, &failed, &failedJoined,
                               outIndexCount, firstWarning, err)) {
            WriteDiagLog("[AddFolder] AddPathsBatchCore FAILED err=" + err);
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }
        WriteDiagLog("[AddFolder] done added=" + std::to_string(added)
                     + " failed=" + std::to_string(failed)
                     + (firstWarning.empty() ? "" : (" warning=" + firstWarning)));
        if (outAddedCount) *outAddedCount = added;
        if (outFailedCount) *outFailedCount = failed;
        if (outFailedPaths && outFailedPathsBufSize > 0 && !failedJoined.empty()) {
            std::strncpy(outFailedPaths, failedJoined.c_str(), outFailedPathsBufSize - 1);
            outFailedPaths[outFailedPathsBufSize - 1] = '\0';
        }

        ok = true;
    } while (false);

    if (ok && !firstWarning.empty())
        SetError(outErrorMsg, errorMsgSize, "warning: " + firstWarning);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_SearchIndex(const char* cadFilePath, int topK,
                                                 char* outIds, int outIdsBufSize,
                                                 float* outScores, int maxResults,
                                                 int* outResultCount,
                                                 char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (!g_embedder) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_LoadEmbeddingsModel was not called");
        return false;
    }
    if (outResultCount) *outResultCount = 0;
    if (outIds && outIdsBufSize > 0) outIds[0] = '\0';

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        std::string err;
        PyObject* vs = GetSharedIndex(err);
        if (!vs) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }

        PyObject* idsObj = PyObject_CallMethod(vs, "get_ids", nullptr);
        if (!idsObj) {
            SetError(outErrorMsg, errorMsgSize, "get_ids failed: " + FetchPythonError());
            break;
        }
        Py_ssize_t existingCount = PySequence_Size(idsObj);
        Py_DECREF(idsObj);
        if (existingCount == 0) {
            // Index is empty: return success with 0 results instead of treating it as an error
            // (same behavior as Web API search_index).
            ok = true;
            break;
        }

        std::vector<float> vec;
        if (!ComputeEmbeddingVector(cadFilePath, vec, err)) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }

        // Part search must exclude assemblies (multi-body files). The FAISS store keeps one row per
        // body but dedups query hits to one per file, so a query for e.g. a nut matches the nut body
        // that lives INSIDE an assembly and surfaces that assembly file. We drop any hit whose record
        // metadata carries kind=="assembly" (kind is written per file at registration time, mirroring
        // the demo_HOOPS_Embeddings_retrieval notebook's filters={"kind":"part"}).
        //
        // A native filters={"kind":"part"} query is NOT used because it strictly excludes records that
        // have no "kind" key, which would return nothing on indexes built before this metadata existed.
        // Instead we over-fetch and post-filter, keeping legacy (kind-less) records for compatibility.
        const int wantK = topK > 0 ? topK : 1;
        int fetchK = wantK * 4;
        if (fetchK < wantK + 32) fetchK = wantK + 32;
        if (fetchK > static_cast<int>(existingCount)) fetchK = static_cast<int>(existingCount);

        PyObject* pyList = PyList_New(static_cast<Py_ssize_t>(vec.size()));
        for (size_t i = 0; i < vec.size(); ++i) {
            PyList_SET_ITEM(pyList, static_cast<Py_ssize_t>(i), PyFloat_FromDouble(vec[i]));
        }
        PyObject* queryMethod = PyObject_GetAttrString(vs, "query");
        if (!queryMethod) {
            Py_DECREF(pyList);
            SetError(outErrorMsg, errorMsgSize, "resolve query failed: " + FetchPythonError());
            break;
        }
        PyObject* argsQuery = Py_BuildValue("(O)", pyList);
        PyObject* kwargsQuery = Py_BuildValue("{s:i}", "top_k", fetchK);
        PyObject* hits = PyObject_Call(queryMethod, argsQuery, kwargsQuery);
        Py_DECREF(argsQuery);
        Py_DECREF(kwargsQuery);
        Py_DECREF(queryMethod);
        Py_DECREF(pyList);
        if (!hits) {
            SetError(outErrorMsg, errorMsgSize, "query failed: " + FetchPythonError());
            break;
        }

        PyObject* seq = PySequence_Fast(hits, "query() result is not a sequence");
        Py_DECREF(hits);
        if (!seq) {
            SetError(outErrorMsg, errorMsgSize, "query() result not iterable: " + FetchPythonError());
            break;
        }

        const int limit = (maxResults < wantK) ? maxResults : wantK;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        int written = 0;
        std::string idsJoined;
        for (Py_ssize_t i = 0; i < n && written < limit; ++i) {
            PyObject* hit = PySequence_Fast_GET_ITEM(seq, i); // borrowed
            PyObject* idObj = PyObject_GetAttrString(hit, "id");
            PyObject* scoreObj = PyObject_GetAttrString(hit, "score");
            if (!idObj || !scoreObj) {
                Py_XDECREF(idObj);
                Py_XDECREF(scoreObj);
                PyErr_Clear();
                continue; // Skip invalid elements
            }

            // Drop assemblies: keep the hit only if its metadata has no kind, or kind != "assembly".
            bool isAssembly = false;
            PyObject* metaObj = PyObject_GetAttrString(hit, "metadata");
            if (metaObj) {
                if (PyDict_Check(metaObj)) {
                    PyObject* kindObj = PyDict_GetItemString(metaObj, "kind"); // borrowed
                    if (kindObj && PyUnicode_Check(kindObj)) {
                        const char* ks = PyUnicode_AsUTF8(kindObj);
                        if (ks && std::strcmp(ks, "assembly") == 0) isAssembly = true;
                    }
                }
                Py_DECREF(metaObj);
            } else {
                PyErr_Clear();
            }
            if (isAssembly) {
                Py_DECREF(idObj);
                Py_DECREF(scoreObj);
                continue;
            }

            const char* idUtf8 = PyUnicode_AsUTF8(idObj);
            const std::string idStr = idUtf8 ? idUtf8 : "";
            const double score = PyFloat_AsDouble(scoreObj);
            Py_DECREF(idObj);
            Py_DECREF(scoreObj);

            if (!idsJoined.empty()) idsJoined += "\n";
            idsJoined += idStr;
            if (outScores) outScores[written] = static_cast<float>(score);
            ++written;
        }
        Py_DECREF(seq);

        if (outIds && outIdsBufSize > 0) {
            std::strncpy(outIds, idsJoined.c_str(), outIdsBufSize - 1);
            outIds[outIdsBufSize - 1] = '\0';
        }
        if (outResultCount) *outResultCount = written;
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

// Bridge driver appended to the embedded AssemblyMatcher source. Exposes two entry points the
// C exports below call through g_asmNs:
//   run_assembly_search(...) -> list[tuple] of (assembly_id, score, geom, coverage, matched,
//                               cand_parts, query_parts)
//   index_counts(store)      -> (files, bodies, assemblies, single_part, dim)
// A per-index matcher cache keyed by (faiss_path, mtime) makes the one-off IDF/k-means build
// (seconds on a large corpus) a first-call-only cost; only the most recent index is retained.
static const char* kAssemblyDriverPy = R"PYDRV(
_matcher_cache = {}

def _get_matcher(embedder, faiss_path):
    import os
    try:
        mtime = os.path.getmtime(faiss_path)
    except OSError:
        mtime = 0.0
    key = (faiss_path, mtime)
    m = _matcher_cache.get(key)
    if m is not None:
        return m
    from hoops_ai.ml import CADSearch
    s = CADSearch(shape_model=embedder)
    b = s.load_shape_index(path=faiss_path)
    m = AssemblyMatcher(searcher=s, embedding_batch=b, min_bodies_for_assembly=2)
    m.build_part_rarity_weights()
    _matcher_cache.clear()   # retain only the most recently used index
    _matcher_cache[key] = m
    return m

def run_assembly_search(embedder, faiss_path, query_path, top_k, candidate_k,
                        sim_thresh, bop_weight, coverage_mode, use_idf):
    m = _get_matcher(embedder, faiss_path)
    results = m.search(query_path=query_path, top_k=int(top_k), candidate_k=int(candidate_k),
                       sim_thresh=float(sim_thresh), method="hungarian", use_idf=bool(use_idf),
                       candidate_mode="search", assemblies_only=True,
                       reuse_index_vectors=True, coverage_mode=str(coverage_mode),
                       bop_weight=float(bop_weight), n_jobs=8)
    out = []
    for r in results:
        out.append((str(r.get("assembly", "")),
                    float(r.get("score", 0.0)),
                    float(r.get("geom_score", 0.0)),
                    float(r.get("coverage", 0.0)),
                    int(r.get("matched", 0)),
                    int(r.get("n_parts", 0)),
                    int(r.get("M", 0))))
    return out

def index_counts(store):
    import faiss
    from collections import Counter
    idm = faiss.vector_to_array(store._index.id_map)
    bodies = int(store._index.ntotal)
    counts = Counter(int(x) for x in idm.tolist())
    files = len(store._id_to_int)
    assemblies = sum(1 for v in counts.values() if v >= 2)
    single = sum(1 for v in counts.values() if v == 1)
    try:
        dim = int(store.dim())
    except Exception:
        dim = 0
    return (files, bodies, assemblies, single, dim)

def part_body_count(store, part_id):
    # Number of body vectors stored for one file id (all bodies of a file share the file path as
    # id). Returns -1 when the id is not registered. >= 2 bodies means the file is an assembly.
    import faiss
    m = store._id_to_int
    if part_id not in m:
        return -1
    target = int(m[part_id])
    idm = faiss.vector_to_array(store._index.id_map)
    return int((idm == target).sum())
)PYDRV";

// Exec the embedded AssemblyMatcher source + driver into g_asmNs exactly once. The caller MUST
// hold the GIL. On success g_asmNs is a live dict namespace exposing run_assembly_search /
// index_counts. Returns false and fills err on failure.
bool EnsureAssemblyModule(std::string& err) {
    if (g_asmNs) return true;
    PyObject* ns = PyDict_New();
    if (!ns) { err = "PyDict_New failed"; return false; }
    // Give the namespace a real __name__ / __builtins__ so imports and class defs behave.
    PyObject* builtins = PyEval_GetBuiltins(); // borrowed
    if (builtins) PyDict_SetItemString(ns, "__builtins__", builtins);
    PyObject* r1 = PyRun_String(kAssemblyMatcherPy, Py_file_input, ns, ns);
    if (!r1) {
        err = "exec AssemblyMatcher source failed: " + FetchPythonError();
        Py_DECREF(ns);
        return false;
    }
    Py_DECREF(r1);
    PyObject* r2 = PyRun_String(kAssemblyDriverPy, Py_file_input, ns, ns);
    if (!r2) {
        err = "exec assembly driver failed: " + FetchPythonError();
        Py_DECREF(ns);
        return false;
    }
    Py_DECREF(r2);
    g_asmNs = ns; // keep alive for the process lifetime
    return true;
}

extern "C" HOOPSAI_API bool HoopsAI_SearchSimilarAssembly(
        const char* cadFilePath, int topK,
        int candidateK, float simThresh, float bopWeight,
        const char* coverageMode, bool useIdf,
        char* outIds, int outIdsBufSize,
        float* outScores, float* outGeomScores, float* outCoverages,
        int* outMatchedParts, int* outCandidateParts, int maxResults,
        int* outResultCount,
        char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (!cadFilePath || !*cadFilePath) {
        SetError(outErrorMsg, errorMsgSize, "cadFilePath is empty");
        return false;
    }
    if (outResultCount) *outResultCount = 0;
    if (outIds && outIdsBufSize > 0) outIds[0] = '\0';

    // Defaults mirror the tutorial's demo_assembly_to_assembly_retrieval workflow.
    const int candK = candidateK > 0 ? candidateK : 30;
    const double simT = simThresh > 0.0f ? static_cast<double>(simThresh) : 0.80;
    const double bopW = bopWeight >= 0.0f ? static_cast<double>(bopWeight) : 0.30;
    std::string covMode = (coverageMode && *coverageMode) ? coverageMode : "symmetric";

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        // Require a current index (the matcher reloads its vectors from the .faiss file).
        if (!g_index || g_indexBase.empty()) {
            SetError(outErrorMsg, errorMsgSize, "no current index: call HoopsAI_OpenIndex first");
            break;
        }
        std::string err;
        if (!EnsureAssemblyModule(err)) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }
        PyObject* fn = PyDict_GetItemString(g_asmNs, "run_assembly_search"); // borrowed
        if (!fn) {
            SetError(outErrorMsg, errorMsgSize, "run_assembly_search not found in assembly module");
            break;
        }
        const std::string faissPath = IndexFaissOf(g_indexBase);
        // embedder is only needed for OUT-of-corpus queries; pass None when no model is loaded so
        // in-corpus queries (reuse of stored body vectors) still work without a model.
        PyObject* embArg = g_embedder ? g_embedder : Py_None;
        PyObject* args = Py_BuildValue("(OssiiddsO)",
                                       embArg,
                                       faissPath.c_str(),
                                       cadFilePath,
                                       topK,
                                       candK,
                                       simT,
                                       bopW,
                                       covMode.c_str(),
                                       useIdf ? Py_True : Py_False);
        if (!args) {
            SetError(outErrorMsg, errorMsgSize, "build args failed: " + FetchPythonError());
            break;
        }
        PyObject* resultObj = PyObject_CallObject(fn, args);
        Py_DECREF(args);
        if (!resultObj) {
            SetError(outErrorMsg, errorMsgSize, "assembly search failed: " + FetchPythonError());
            break;
        }
        PyObject* seq = PySequence_Fast(resultObj, "assembly search result is not a sequence");
        Py_DECREF(resultObj);
        if (!seq) {
            SetError(outErrorMsg, errorMsgSize, "assembly search result not iterable: " + FetchPythonError());
            break;
        }
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        int written = 0;
        std::string idsJoined;
        for (Py_ssize_t i = 0; i < n && written < maxResults; ++i) {
            PyObject* row = PySequence_Fast_GET_ITEM(seq, i); // borrowed; expect a 7-tuple
            if (!PySequence_Check(row) || PySequence_Size(row) < 7) continue;
            PyObject* idObj    = PySequence_GetItem(row, 0);
            PyObject* scoreObj = PySequence_GetItem(row, 1);
            PyObject* geomObj  = PySequence_GetItem(row, 2);
            PyObject* covObj   = PySequence_GetItem(row, 3);
            PyObject* matchObj = PySequence_GetItem(row, 4);
            PyObject* candObj  = PySequence_GetItem(row, 5);
            const char* idUtf8 = idObj ? PyUnicode_AsUTF8(idObj) : nullptr;
            const std::string idStr = idUtf8 ? idUtf8 : "";
            if (!idsJoined.empty()) idsJoined += "\n";
            idsJoined += idStr;
            if (outScores)        outScores[written]        = static_cast<float>(PyFloat_AsDouble(scoreObj));
            if (outGeomScores)    outGeomScores[written]    = static_cast<float>(PyFloat_AsDouble(geomObj));
            if (outCoverages)     outCoverages[written]     = static_cast<float>(PyFloat_AsDouble(covObj));
            if (outMatchedParts)  outMatchedParts[written]  = static_cast<int>(PyLong_AsLong(matchObj));
            if (outCandidateParts)outCandidateParts[written]= static_cast<int>(PyLong_AsLong(candObj));
            Py_XDECREF(idObj); Py_XDECREF(scoreObj); Py_XDECREF(geomObj);
            Py_XDECREF(covObj); Py_XDECREF(matchObj); Py_XDECREF(candObj);
            ++written;
        }
        Py_DECREF(seq);
        if (outIds && outIdsBufSize > 0) {
            std::strncpy(outIds, idsJoined.c_str(), outIdsBufSize - 1);
            outIds[outIdsBufSize - 1] = '\0';
        }
        if (outResultCount) *outResultCount = written;
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

// ---------------------------------------------------------------------------
// Shape map (cluster visualization) support.
// The driver reads the CURRENT index's stored per-body vectors through the same public path the
// assembly matcher uses (CADSearch.load_shape_index -> EmbeddingBatch.ids/.values), aggregates
// them to one vector per file id, projects to 2D/3D via PCA (numpy SVD), and clusters with FAISS
// k-means. No CAD re-embedding: this scales to tens of thousands of files.
static const char* kShapeMapPy = R"PYSMAP(
import numpy as np

def _l2norm_rows(mat):
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n

def compute_shape_map(embedder, faiss_path, n_clusters, dims, only_ids=None):
    from hoops_ai.ml import CADSearch
    from collections import OrderedDict
    s = CADSearch(shape_model=embedder)
    b = s.load_shape_index(path=faiss_path)
    ids = list(b.ids)
    vals = _l2norm_rows(b.values)                      # (n_bodies, dim), normalized

    # Aggregate per file id: mean of that id's body rows, then renormalize -> one point per file.
    groups = OrderedDict()
    for row, fid in enumerate(ids):
        groups.setdefault(fid, []).append(row)
    file_ids = list(groups.keys())

    # Optional subset filter: when only_ids is given (a list/set of file ids), keep only those
    # files so the projection and clustering are computed on the subset alone. This is how the
    # caller restricts the map to, e.g., tagged parts. Ids not present in the index are ignored.
    if only_ids:
        allow = set(str(x) for x in only_ids)
        file_ids = [fid for fid in file_ids if str(fid) in allow]

    N = len(file_ids)
    if N == 0:
        return ([], 0)
    agg = np.zeros((N, vals.shape[1]), dtype=np.float32)
    for i, fid in enumerate(file_ids):
        agg[i] = vals[groups[fid]].mean(axis=0)
    agg = _l2norm_rows(agg)

    d = 3 if int(dims) >= 3 else 2
    d = max(1, min(d, agg.shape[1], N))

    # PCA via economy SVD; rows of Vt are the principal directions.
    mean = agg.mean(axis=0, keepdims=True)
    X = agg - mean
    try:
        _U, _S, Vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ Vt[:d].T
    except Exception:
        coords = X[:, :d]
    coords = np.asarray(coords, dtype=np.float32)

    # Scale into a stable [-1, 1] cube for a fixed render scale.
    span = float(np.max(np.abs(coords))) if coords.size else 1.0
    if span <= 0.0:
        span = 1.0
    coords = coords / span
    if coords.shape[1] < 3:
        coords = np.concatenate(
            [coords, np.zeros((N, 3 - coords.shape[1]), dtype=np.float32)], axis=1)

    # Cluster the aggregated vectors with FAISS k-means (fast even for large N).
    k = int(n_clusters) if int(n_clusters) > 0 else min(max(2, N // 50), 24)
    k = max(1, min(k, N))
    if k <= 1:
        assign = np.zeros(N, dtype=np.int64)
    else:
        import faiss
        Xk = np.ascontiguousarray(agg, dtype=np.float32)
        km = faiss.Kmeans(Xk.shape[1], k, niter=20, seed=42, verbose=False)
        km.train(Xk)
        _, a = km.index.search(Xk, 1)
        assign = a.reshape(-1).astype(np.int64)

    out = []
    for i, fid in enumerate(file_ids):
        out.append((str(fid), float(coords[i, 0]), float(coords[i, 1]),
                    float(coords[i, 2]), int(assign[i])))
    return (out, int(k))
)PYSMAP";

// Exec the shape-map driver into g_shapeMapNs exactly once. The caller MUST hold the GIL.
static bool EnsureShapeMapModule(std::string& err) {
    if (g_shapeMapNs) return true;
    PyObject* ns = PyDict_New();
    if (!ns) { err = "PyDict_New failed"; return false; }
    PyObject* builtins = PyEval_GetBuiltins(); // borrowed
    if (builtins) PyDict_SetItemString(ns, "__builtins__", builtins);
    PyObject* r = PyRun_String(kShapeMapPy, Py_file_input, ns, ns);
    if (!r) {
        err = "exec shape-map driver failed: " + FetchPythonError();
        Py_DECREF(ns);
        return false;
    }
    Py_DECREF(r);
    g_shapeMapNs = ns; // keep alive for the process lifetime
    return true;
}

extern "C" HOOPSAI_API bool HoopsAI_ComputeIndexShapeMap(int nClusters, int dims,
                                                          char const* filterIds,
                                                          char* outIds, int outIdsBufSize,
                                                          float* outCoords, int* outClusterIds,
                                                          int maxPoints, int* outPointCount,
                                                          int* outClusterCount,
                                                          char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (outPointCount) *outPointCount = 0;
    if (outClusterCount) *outClusterCount = 0;
    if (outIds && outIdsBufSize > 0) outIds[0] = '\0';

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        if (!g_index || g_indexBase.empty()) {
            SetError(outErrorMsg, errorMsgSize, "no current index: call HoopsAI_OpenIndex first");
            break;
        }
        std::string err;
        if (!EnsureShapeMapModule(err)) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }
        PyObject* fn = PyDict_GetItemString(g_shapeMapNs, "compute_shape_map"); // borrowed
        if (!fn) {
            SetError(outErrorMsg, errorMsgSize, "compute_shape_map not found in shape-map module");
            break;
        }
        const std::string faissPath = IndexFaissOf(g_indexBase);
        PyObject* embArg = g_embedder ? g_embedder : Py_None; // not needed to read stored vectors

        // Build the optional id filter as a Python list of str (or None for the whole index).
        // filterIds is a newline-delimited UTF-8 list; blank lines are skipped.
        PyObject* idsFilter = nullptr;
        if (filterIds && filterIds[0] != '\0') {
            idsFilter = PyList_New(0);
            if (idsFilter) {
                std::string all(filterIds);
                size_t start = 0;
                while (start <= all.size()) {
                    size_t nl = all.find('\n', start);
                    std::string line = all.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (!line.empty()) {
                        PyObject* s = PyUnicode_FromString(line.c_str());
                        if (s) { PyList_Append(idsFilter, s); Py_DECREF(s); }
                    }
                    if (nl == std::string::npos) break;
                    start = nl + 1;
                }
            }
        }
        if (!idsFilter) { idsFilter = Py_None; Py_INCREF(idsFilter); }

        PyObject* args = Py_BuildValue("(OsiiO)", embArg, faissPath.c_str(), nClusters, dims, idsFilter);
        Py_DECREF(idsFilter); // args holds its own reference via "O"
        if (!args) {
            SetError(outErrorMsg, errorMsgSize, "build args failed: " + FetchPythonError());
            break;
        }
        PyObject* resultObj = PyObject_CallObject(fn, args);
        Py_DECREF(args);
        if (!resultObj) {
            SetError(outErrorMsg, errorMsgSize, "compute_shape_map failed: " + FetchPythonError());
            break;
        }
        PyObject* rowsObj = nullptr;
        int kClusters = 0;
        if (!PyArg_ParseTuple(resultObj, "Oi", &rowsObj, &kClusters)) { // rowsObj borrowed
            Py_DECREF(resultObj);
            SetError(outErrorMsg, errorMsgSize, "compute_shape_map returned an unexpected shape: " + FetchPythonError());
            break;
        }
        PyObject* seq = PySequence_Fast(rowsObj, "shape-map result is not a sequence");
        if (!seq) {
            Py_DECREF(resultObj);
            SetError(outErrorMsg, errorMsgSize, "shape-map result not iterable: " + FetchPythonError());
            break;
        }
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        int written = 0;
        std::string idsJoined;
        for (Py_ssize_t i = 0; i < n && written < maxPoints; ++i) {
            PyObject* row = PySequence_Fast_GET_ITEM(seq, i); // borrowed; expect a 5-tuple
            if (!PySequence_Check(row) || PySequence_Size(row) < 5) continue;
            PyObject* idObj = PySequence_GetItem(row, 0);
            PyObject* xObj  = PySequence_GetItem(row, 1);
            PyObject* yObj  = PySequence_GetItem(row, 2);
            PyObject* zObj  = PySequence_GetItem(row, 3);
            PyObject* cObj  = PySequence_GetItem(row, 4);
            const char* idUtf8 = idObj ? PyUnicode_AsUTF8(idObj) : nullptr;
            if (!idsJoined.empty()) idsJoined += "\n";
            idsJoined += (idUtf8 ? idUtf8 : "");
            if (outCoords) {
                outCoords[written * 3 + 0] = static_cast<float>(PyFloat_AsDouble(xObj));
                outCoords[written * 3 + 1] = static_cast<float>(PyFloat_AsDouble(yObj));
                outCoords[written * 3 + 2] = static_cast<float>(PyFloat_AsDouble(zObj));
            }
            if (outClusterIds) outClusterIds[written] = static_cast<int>(PyLong_AsLong(cObj));
            Py_XDECREF(idObj); Py_XDECREF(xObj); Py_XDECREF(yObj); Py_XDECREF(zObj); Py_XDECREF(cObj);
            ++written;
        }
        Py_DECREF(seq);
        Py_DECREF(resultObj);
        if (outIds && outIdsBufSize > 0) {
            std::strncpy(outIds, idsJoined.c_str(), outIdsBufSize - 1);
            outIds[outIdsBufSize - 1] = '\0';
        }
        if (outPointCount) *outPointCount = written;
        if (outClusterCount) *outClusterCount = kClusters;
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_GetIndexStats(char* outPath, int outPathBufSize,
                                                   bool* outHasIndex,
                                                   int* outFileCount, int* outBodyCount,
                                                   int* outAssemblyCount, int* outSinglePartCount,
                                                   int* outDim,
                                                   char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (outHasIndex) *outHasIndex = false;
    if (outFileCount) *outFileCount = 0;
    if (outBodyCount) *outBodyCount = 0;
    if (outAssemblyCount) *outAssemblyCount = 0;
    if (outSinglePartCount) *outSinglePartCount = 0;
    if (outDim) *outDim = 0;
    if (outPath && outPathBufSize > 0) outPath[0] = '\0';

    // No current index: not an error; report absence via outHasIndex (same style as GetCurrentIndexInfo).
    if (!g_index) return true;

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        std::string err;
        if (!EnsureAssemblyModule(err)) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }
        PyObject* fn = PyDict_GetItemString(g_asmNs, "index_counts"); // borrowed
        if (!fn) {
            SetError(outErrorMsg, errorMsgSize, "index_counts not found in assembly module");
            break;
        }
        PyObject* res = PyObject_CallFunctionObjArgs(fn, g_index, nullptr);
        if (!res) {
            SetError(outErrorMsg, errorMsgSize, "index_counts failed: " + FetchPythonError());
            break;
        }
        int files = 0, bodies = 0, asm_ = 0, single = 0, dim = 0;
        if (!PyArg_ParseTuple(res, "iiiii", &files, &bodies, &asm_, &single, &dim)) {
            Py_DECREF(res);
            SetError(outErrorMsg, errorMsgSize, "index_counts returned an unexpected shape: " + FetchPythonError());
            break;
        }
        Py_DECREF(res);
        if (outHasIndex) *outHasIndex = true;
        if (outFileCount) *outFileCount = files;
        if (outBodyCount) *outBodyCount = bodies;
        if (outAssemblyCount) *outAssemblyCount = asm_;
        if (outSinglePartCount) *outSinglePartCount = single;
        // Use the bridge's tracked dimension (set from the .faiss file at open time and valid even
        // with no model loaded); the store's own dim() is not reliably populated here. `dim` from
        // index_counts is ignored.
        (void)dim;
        if (outDim) *outDim = g_indexDim;
        if (outPath && outPathBufSize > 0) {
            const std::string faissPath = IndexFaissOf(g_indexBase);
            std::strncpy(outPath, faissPath.c_str(), outPathBufSize - 1);
            outPath[outPathBufSize - 1] = '\0';
        }
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_GetPartBodyCount(const char* partId,
                                                      int* outBodyCount, bool* outIsAssembly,
                                                      char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (outBodyCount) *outBodyCount = 0;
    if (outIsAssembly) *outIsAssembly = false;

    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    const std::string id = (partId && *partId) ? partId : "";
    if (id.empty()) {
        SetError(outErrorMsg, errorMsgSize, "partId is empty");
        return false;
    }
    if (!g_index) {
        SetError(outErrorMsg, errorMsgSize, "no current index: call HoopsAI_OpenIndex first");
        return false;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        std::string err;
        if (!EnsureAssemblyModule(err)) {
            SetError(outErrorMsg, errorMsgSize, err);
            break;
        }
        PyObject* fn = PyDict_GetItemString(g_asmNs, "part_body_count"); // borrowed
        if (!fn) {
            SetError(outErrorMsg, errorMsgSize, "part_body_count not found in assembly module");
            break;
        }
        PyObject* idObj = PyUnicode_FromString(id.c_str());
        PyObject* res = PyObject_CallFunctionObjArgs(fn, g_index, idObj, nullptr);
        Py_XDECREF(idObj);
        if (!res) {
            SetError(outErrorMsg, errorMsgSize, "part_body_count failed: " + FetchPythonError());
            break;
        }
        long const count = PyLong_AsLong(res);
        Py_DECREF(res);
        if (count == -1 && PyErr_Occurred()) {
            SetError(outErrorMsg, errorMsgSize, "part_body_count returned a non-integer: " + FetchPythonError());
            break;
        }
        if (count < 0) {
            SetError(outErrorMsg, errorMsgSize, "part is not registered in the current index");
            break;
        }
        if (outBodyCount) *outBodyCount = static_cast<int>(count);
        if (outIsAssembly) *outIsAssembly = (count >= 2);
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_OpenIndex(const char* faissFilePath, bool createIfMissing,
                                               char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    const std::string reqPath = (faissFilePath && *faissFilePath) ? NormalizeSlashes(faissFilePath) : "";
    if (reqPath.empty()) {
        SetError(outErrorMsg, errorMsgSize, "faissFilePath is empty (the client must supply a .faiss file path)");
        return false;
    }
    // The client selects a .faiss file; derive the base path (<dir>/<stem>) from it.
    const std::string basePath = IndexBaseFromFaiss(reqPath); // <dir>/<stem> (no extension)
    // Same style as loading a ckpt: opening the already-current index is a no-op.
    if (g_index && basePath == g_indexBase) return true;

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        const std::string faissPath = IndexFaissOf(basePath); // <base>.faiss
        std::error_code ec;
        const bool exists = fs::exists(U8Path(faissPath), ec);

        if (!exists && !createIfMissing) {
            SetError(outErrorMsg, errorMsgSize,
                     "index not found and createIfMissing is false: " + faissPath);
            break;
        }

        PyObject* embMod = PyImport_ImportModule("hoops_ai.ml.embeddings");
        if (!embMod) {
            SetError(outErrorMsg, errorMsgSize,
                     "import hoops_ai.ml.embeddings failed: " + FetchPythonError());
            break;
        }
        PyObject* storeClass = PyObject_GetAttrString(embMod, "FaissVectorStore");
        Py_DECREF(embMod);
        if (!storeClass) {
            SetError(outErrorMsg, errorMsgSize,
                     "resolve FaissVectorStore failed: " + FetchPythonError());
            break;
        }

        PyObject* vs = nullptr;
        int newDim = 0;
        if (exists) {
            // FaissVectorStore.load(base) is a class method returning a new instance.
            vs = PyObject_CallMethod(storeClass, "load", "(s)", basePath.c_str());
            if (!vs) {
                SetError(outErrorMsg, errorMsgSize,
                         "FaissVectorStore.load failed: " + FetchPythonError());
                Py_DECREF(storeClass);
                break;
            }
            // Determine the index dimension from the loaded store.
            PyObject* dimObj = PyObject_GetAttrString(vs, "dim");
            if (dimObj && PyLong_Check(dimObj)) {
                newDim = static_cast<int>(PyLong_AsLong(dimObj));
            } else {
                PyErr_Clear();
            }
            Py_XDECREF(dimObj);
            // If a model is loaded, its dimension must match the existing index.
            if (g_embedder && newDim > 0) {
                const int modelDim = DeriveEmbeddingDim();
                if (modelDim != newDim) {
                    SetError(outErrorMsg, errorMsgSize,
                             "embeddings model dimension (" + std::to_string(modelDim) +
                             ") does not match index dimension (" + std::to_string(newDim) + ")");
                    Py_DECREF(vs);
                    Py_DECREF(storeClass);
                    break;
                }
            }
        } else {
            // Creating a new index requires a loaded model to derive the dimension.
            if (!g_embedder) {
                SetError(outErrorMsg, errorMsgSize,
                         "load an embeddings model first (the index dimension is derived from it)");
                Py_DECREF(storeClass);
                break;
            }
            // Create the parent folder (recursively) and the per-index folder (<base>/,
            // named after the index) before the first save. The .faiss / .meta files are
            // written next to that folder by SaveIndexAtomic below.
            std::string mkErr;
            if (!EnsureDir(U8Path(basePath).parent_path(), mkErr)) {
                SetError(outErrorMsg, errorMsgSize, "create index directory failed: " + mkErr);
                Py_DECREF(storeClass);
                break;
            }
            if (!EnsureDir(U8Path(ThumbnailsDirOf(basePath)), mkErr)) {
                SetError(outErrorMsg, errorMsgSize, "create index folder failed: " + mkErr);
                Py_DECREF(storeClass);
                break;
            }
            newDim = DeriveEmbeddingDim();
            vs = PyObject_CallFunction(storeClass, "(i)", newDim);
            if (!vs) {
                SetError(outErrorMsg, errorMsgSize,
                         "construct FaissVectorStore failed: " + FetchPythonError());
                Py_DECREF(storeClass);
                break;
            }
            std::string saveErr;
            if (!SaveIndexAtomic(vs, basePath, saveErr)) {
                SetError(outErrorMsg, errorMsgSize,
                         "FaissVectorStore.save (initial) failed: " + saveErr);
                Py_DECREF(vs);
                Py_DECREF(storeClass);
                break;
            }
        }
        Py_DECREF(storeClass);

        // Swap the current index only on success.
        CloseCurrentIndex();
        g_index = vs; // Keep for the lifetime of the current index (do not transfer refcount)
        g_indexBase = basePath;
        g_indexDim = (newDim > 0) ? newDim : kIndexDim;
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_CloseIndex(char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    CloseCurrentIndex(); // in-memory release only; never deletes files
    PyGILState_Release(gstate);
    return true;
}

extern "C" HOOPSAI_API bool HoopsAI_GetCurrentIndexInfo(char* outPath, int outPathBufSize,
                                                        bool* outHasIndex,
                                                        int* outCount, int* outDim,
                                                        char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (outHasIndex) *outHasIndex = false;
    if (outCount) *outCount = 0;
    if (outDim) *outDim = 0;
    if (outPath && outPathBufSize > 0) outPath[0] = '\0';

    // No current index: not an error; report absence via outHasIndex.
    if (!g_index) return true;

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        PyObject* idsObj = PyObject_CallMethod(g_index, "get_ids", nullptr);
        if (!idsObj) {
            SetError(outErrorMsg, errorMsgSize, "get_ids failed: " + FetchPythonError());
            break;
        }
        Py_ssize_t count = PySequence_Size(idsObj);
        Py_DECREF(idsObj);

        if (outHasIndex) *outHasIndex = true;
        if (outCount) *outCount = static_cast<int>(count);
        if (outDim) *outDim = g_indexDim;
        if (outPath && outPathBufSize > 0) {
            const std::string faissPath = IndexFaissOf(g_indexBase);
            std::strncpy(outPath, faissPath.c_str(), outPathBufSize - 1);
            outPath[outPathBufSize - 1] = '\0';
        }
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_ListIndexParts(char* outIds, int outIdsBufSize,
                                                   int limit, int* outResultCount,
                                                   char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    if (outResultCount) *outResultCount = 0;
    if (outIds && outIdsBufSize > 0) outIds[0] = '\0';

    // No current index: not an error; report zero results (same style as GetCurrentIndexInfo).
    if (!g_index) return true;

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        PyObject* idsObj = PyObject_CallMethod(g_index, "get_ids", nullptr);
        if (!idsObj) {
            SetError(outErrorMsg, errorMsgSize, "get_ids failed: " + FetchPythonError());
            break;
        }
        PyObject* seq = PySequence_Fast(idsObj, "get_ids() result is not a sequence");
        Py_DECREF(idsObj);
        if (!seq) {
            SetError(outErrorMsg, errorMsgSize, "get_ids() result not iterable: " + FetchPythonError());
            break;
        }

        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        int written = 0;
        std::string idsJoined;
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (limit > 0 && written >= limit) break;
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i); // borrowed
            const char* idUtf8 = PyUnicode_AsUTF8(item);
            if (!idUtf8) {
                PyErr_Clear();
                continue; // Skip non-string / undecodable IDs defensively.
            }
            if (!idsJoined.empty()) idsJoined += "\n";
            idsJoined += idUtf8;
            ++written;
        }
        Py_DECREF(seq);

        if (outIds && outIdsBufSize > 0) {
            std::strncpy(outIds, idsJoined.c_str(), outIdsBufSize - 1);
            outIds[outIdsBufSize - 1] = '\0';
        }
        if (outResultCount) *outResultCount = written;
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_GetPartThumbnailPath(const char* partId,
                                                         char* outPath, int outPathBufSize,
                                                         bool* outExists,
                                                         char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (outExists) *outExists = false;
    if (outPath && outPathBufSize > 0) outPath[0] = '\0';

    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    const std::string id = (partId && *partId) ? partId : "";
    if (id.empty()) {
        SetError(outErrorMsg, errorMsgSize, "partId is empty");
        return false;
    }
    if (!g_index || g_indexBase.empty()) {
        SetError(outErrorMsg, errorMsgSize, "no current index: call HoopsAI_OpenIndex first");
        return false;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        // The part must be registered in the current index.
        PyObject* idsObj = PyObject_CallMethod(g_index, "get_ids", nullptr);
        if (!idsObj) {
            SetError(outErrorMsg, errorMsgSize, "get_ids failed: " + FetchPythonError());
            break;
        }
        PyObject* pyId = PyUnicode_FromString(id.c_str());
        int contains = PySequence_Contains(idsObj, pyId);
        Py_DECREF(pyId);
        Py_DECREF(idsObj);
        if (contains < 0) {
            SetError(outErrorMsg, errorMsgSize, "membership check failed: " + FetchPythonError());
            break;
        }
        if (contains == 0) {
            SetError(outErrorMsg, errorMsgSize, "part not registered in the current index: " + id);
            break;
        }

        // Resolve the thumbnail. Prefer the relative path stored in the record's metadata
        // (authoritative: it is exactly what hoops_ai generate_images wrote, e.g.
        // "<parent>/<stem>_white.png", and also covers legacy "<stem>.png" records since it is
        // resolved against the same per-index folder). Fall back to the deterministic
        // generate_images rule, then to the legacy flat name, so Web API / older indexes still
        // resolve. All candidates are joined under the per-index images folder.
        const std::string imgDir = ThumbnailsDirOf(g_indexBase);
        std::vector<std::string> candidates;
        auto addCandidate = [&](const std::string& rel) {
            if (rel.empty()) return;
            candidates.push_back(
                NormalizeSlashes((U8Path(imgDir) / U8Path(rel)).u8string()));
        };
        addCandidate(LookupThumbFromMetadata(g_index, id)); // stored relative thumbnail
        addCandidate(GenImagesRelThumb(id));                // <parent>/<stem>_white.png
        addCandidate(ThumbnailNameFor(id));                 // legacy <stem>.png
        if (candidates.empty()) {
            SetError(outErrorMsg, errorMsgSize, "failed to derive thumbnail name for: " + id);
            break;
        }

        std::error_code ec;
        std::string thumbPath = candidates.front();
        bool present = false;
        for (const std::string& c : candidates) {
            if (fs::exists(U8Path(c), ec)) { thumbPath = c; present = true; break; }
        }
        if (outExists) *outExists = present;

        const int needed = static_cast<int>(thumbPath.size()) + 1; // incl. NUL
        if (outPath) {
            if (outPathBufSize < needed) {
                SetError(outErrorMsg, errorMsgSize,
                         "outPath too small: need " + std::to_string(needed) +
                         " bytes, have " + std::to_string(outPathBufSize));
                break;
            }
            std::strncpy(outPath, thumbPath.c_str(), outPathBufSize - 1);
            outPath[outPathBufSize - 1] = '\0';
        }
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API bool HoopsAI_ListIndexPartsPaged(int offset, int count,
                                                        char* outIds, int outIdsBufSize,
                                                        char* outThumbs, int outThumbsBufSize,
                                                        unsigned char* outExists, int maxResults,
                                                        int* outResultCount, int* outTotalCount,
                                                        char* outErrorMsg, int errorMsgSize) {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (outResultCount) *outResultCount = 0;
    if (outTotalCount) *outTotalCount = 0;
    if (outIds && outIdsBufSize > 0) outIds[0] = '\0';
    if (outThumbs && outThumbsBufSize > 0) outThumbs[0] = '\0';

    if (!g_initialized) {
        SetError(outErrorMsg, errorMsgSize, "HoopsAI_Initialize was not called");
        return false;
    }
    // No current index: not an error; report zero results / zero total.
    if (!g_index || g_indexBase.empty()) return true;

    PyGILState_STATE gstate = PyGILState_Ensure();
    bool ok = false;

    do {
        PyObject* idsObj = PyObject_CallMethod(g_index, "get_ids", nullptr);
        if (!idsObj) {
            SetError(outErrorMsg, errorMsgSize, "get_ids failed: " + FetchPythonError());
            break;
        }
        PyObject* seq = PySequence_Fast(idsObj, "get_ids() result is not a sequence");
        Py_DECREF(idsObj);
        if (!seq) {
            SetError(outErrorMsg, errorMsgSize, "get_ids() result not iterable: " + FetchPythonError());
            break;
        }
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        if (outTotalCount) *outTotalCount = static_cast<int>(n);

        // Build id -> relative-thumbnail map in ONE metadata pass so per-id resolution below is
        // O(1). Doing a LookupThumbFromMetadata per id would re-scan all metadata each time and
        // reintroduce the O(n^2) cost this paged API exists to avoid.
        std::unordered_map<std::string, std::string> thumbRelById;
        {
            PyObject* metaIter = PyObject_CallMethod(g_index, "iter_metadata", nullptr);
            if (metaIter) {
                PyObject* iter = PyObject_GetIter(metaIter);
                Py_DECREF(metaIter);
                if (iter) {
                    PyObject* item = nullptr;
                    while ((item = PyIter_Next(iter)) != nullptr) {
                        if (PyDict_Check(item)) {
                            PyObject* fid = PyDict_GetItemString(item, "file_id"); // borrowed
                            if (!fid) fid = PyDict_GetItemString(item, "id");      // borrowed
                            PyObject* th = PyDict_GetItemString(item, "thumbnail"); // borrowed
                            if (fid && PyUnicode_Check(fid) && th && PyUnicode_Check(th)) {
                                const char* f = PyUnicode_AsUTF8(fid);
                                const char* t = PyUnicode_AsUTF8(th);
                                if (f && t) thumbRelById.emplace(std::string(f), std::string(t));
                            }
                        }
                        Py_DECREF(item);
                    }
                    Py_DECREF(iter);
                }
            }
            if (PyErr_Occurred()) PyErr_Clear();
        }

        const std::string imgDir = ThumbnailsDirOf(g_indexBase);
        auto joinImg = [&](const std::string& rel) -> std::string {
            return NormalizeSlashes((U8Path(imgDir) / U8Path(rel)).u8string());
        };

        // Clamp the requested window to [0, n], honoring count (<=0 => to end) and maxResults.
        Py_ssize_t start = offset > 0 ? static_cast<Py_ssize_t>(offset) : 0;
        if (start > n) start = n;
        Py_ssize_t end = (count > 0) ? (start + static_cast<Py_ssize_t>(count)) : n;
        if (end > n) end = n;
        if (maxResults > 0 && end > start + static_cast<Py_ssize_t>(maxResults))
            end = start + static_cast<Py_ssize_t>(maxResults);

        std::string idsJoined, thumbsJoined;
        int written = 0;
        std::error_code ec;
        for (Py_ssize_t i = start; i < end; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i); // borrowed
            const char* idUtf8 = PyUnicode_AsUTF8(item);
            if (!idUtf8) { PyErr_Clear(); continue; } // skip undecodable ids defensively
            const std::string id = idUtf8;

            // Resolve the thumbnail: stored metadata (O(1) via map) -> gen-images rule -> legacy
            // flat name. First candidate that exists on disk wins; otherwise keep the first.
            std::string thumbPath;
            bool present = false;
            std::string candidates[3];
            int nc = 0;
            auto it = thumbRelById.find(id);
            if (it != thumbRelById.end() && !it->second.empty()) candidates[nc++] = joinImg(it->second);
            { const std::string g = GenImagesRelThumb(id); if (!g.empty()) candidates[nc++] = joinImg(g); }
            { const std::string l = ThumbnailNameFor(id);  if (!l.empty()) candidates[nc++] = joinImg(l); }
            if (nc > 0) {
                thumbPath = candidates[0];
                for (int c = 0; c < nc; ++c) {
                    if (fs::exists(U8Path(candidates[c]), ec)) { thumbPath = candidates[c]; present = true; break; }
                }
            }

            if (!idsJoined.empty()) idsJoined += "\n";
            idsJoined += id;
            if (!thumbsJoined.empty()) thumbsJoined += "\n";
            thumbsJoined += thumbPath; // may be empty; parallel to ids by index
            if (outExists && written < maxResults) outExists[written] = present ? 1 : 0;
            ++written;
        }
        Py_DECREF(seq);

        if (outIds && outIdsBufSize > 0) {
            std::strncpy(outIds, idsJoined.c_str(), outIdsBufSize - 1);
            outIds[outIdsBufSize - 1] = '\0';
        }
        if (outThumbs && outThumbsBufSize > 0) {
            std::strncpy(outThumbs, thumbsJoined.c_str(), outThumbsBufSize - 1);
            outThumbs[outThumbsBufSize - 1] = '\0';
        }
        if (outResultCount) *outResultCount = written;
        ok = true;
    } while (false);

    PyGILState_Release(gstate);
    return ok;
}

extern "C" HOOPSAI_API void HoopsAI_Shutdown() {
    std::lock_guard<std::mutex> lock(g_pyMutex);
    if (g_initialized) {
        // Re-acquire the GIL that HoopsAI_Initialize released via PyEval_SaveThread(),
        // then finalize. After this the interpreter is gone, so clear the saved state.
        if (g_mainThreadState) {
            PyEval_RestoreThread(g_mainThreadState);
            g_mainThreadState = nullptr;
        }
        // Python objects held globally are released by Py_FinalizeEx,
        // so only reset the pointers here without explicit DECREF.
        g_sharedCadLoader = nullptr;
        g_embedder = nullptr;
        g_mfrModel = nullptr;
        g_index = nullptr; // release the current index (files on disk are never deleted)
        g_asmNs = nullptr; // assembly-search namespace (its matcher cache) is freed by Py_FinalizeEx
        g_indexBase.clear();
        g_embedderCkptPath.clear();
        g_mfrCkptPath.clear();
        g_currentEmbModelName = kEmbeddingsModelName;
        g_embModelSeq = 0;
        g_indexDim = 0;
        Py_FinalizeEx();
        g_initialized = false;
    }
}
