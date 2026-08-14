// hoops_ai_bridge.h
// C ABI for calling HOOPS AI (Python/hoops_ai) through a local library
// from native applications on Windows and from Linux console applications.
//
// Design principles:
//   - Embed exactly one CPython interpreter inside the library
//   - Import hoops_ai (a Nuitka-compiled .pyd/.so) as a normal Python module
//   - Callers only need to interact with the C ABI and do not need to be aware of Python
//   - Share CAD ACCESS (HOOPSLoader creation) across multiple features such as MFR and Embeddings
//     (see GetSharedCADLoader() in the .cpp for the internal implementation)
//
// On Windows this is built as hoops_ai_bridge.dll, and on Linux as libhoops_ai_bridge.so.
// hoops_ai supports both Windows (x86_64) and GNU/Linux (x86_64, arm64-v8a)
// (Supported Platforms: https://docs.techsoft3d.com/hoops/ai/getting_started/supported_platforms.html)
// so this bridge is also designed to be buildable cross-platform.
//
// [Important design note]
// This bridge is a reference implementation that demonstrates the HOOPS AI invocation mechanism.
// These exported functions (such as HoopsAI_RunMFRInference / HoopsAI_LoadEmbeddingsModel /
// HoopsAI_ComputeEmbedding) are not intended to be exposed directly as your product's public API
// or as an end-user-facing plugin interface. Rather than simply wrapping HOOPS AI tutorial features
// and samples, the expectation is that partners will build a value-added module on top of them
// (for example, constructing and running inference with an end-user-specific trained model based on
// the company's own CAD history data, or connecting to an internal knowledge base to predict cost
// or machining processes).
// When turning this into a production product, at minimum the ISV application layer should:
//   - Fix and bundle checkpointPath during build/installation, and do not expose it at runtime
//     as a parameter that allows arbitrary external paths
//   - Interpret raw label IDs returned by MFR inference in the ISV's business logic
//     (for example, as "chamfer", "hole", etc.) before showing them in the end-user UI
//   - Design exported function names and arguments to match your product features, rather than
//     exposing HOOPS AI internal concepts as-is
#pragma once

#if defined(_WIN32)
    #ifdef HOOPSAI_BRIDGE_EXPORTS
        #define HOOPSAI_API __declspec(dllexport)
    #else
        #define HOOPSAI_API __declspec(dllimport)
    #endif
#else
    #define HOOPSAI_API __attribute__((visibility("default")))
#endif

extern "C" {

    // --- Initialization / shutdown (once per process) ---
    //
    // [Thread-safety] Every exported function below is thread-safe: calls are serialized
    // internally by a single process-wide lock plus the Python GIL, so overlapping calls
    // from different threads simply wait their turn (they never run truly concurrently).
    // HoopsAI_Initialize releases the GIL once start-up finishes, which is what lets the
    // other functions be invoked from a background/worker thread (e.g. a Qt worker) without
    // deadlocking or freezing the caller's UI. HoopsAI_Initialize / HoopsAI_Shutdown must be
    // called once per process (both are idempotent: a second Initialize or Shutdown is a
    // no-op) and should be paired on the same overall lifetime.

    // Initialize the CPython interpreter inside the library, import hoops_ai, and set the license.
    // venvSitePackages: absolute path (UTF-8) to a venv site-packages directory that
    //                    contains hoops_ai, e.g. "C:\\SDK\\HOOPS_AI\\V1.1\\.venv\\Lib\\site-packages"
    //                    (Windows) or "/opt/hoops_ai/.venv/lib/python3.12/site-packages" (Linux).
    //                    The bridge neither validates nor auto-detects this path; it simply adds
    //                    it to sys.path as-is. Resolving the location is the caller's responsibility
    //                    (HOOPS_AI_HOME in development, <exe>\\..\\.venv in a redistribution layout).
    //                    Passing nullptr/empty adds nothing to sys.path.
    // pythonHome:        Python installation used as the base for the venv
    //                    (intended for Windows only; on Linux, nullptr is usually fine).
    //                    On Windows, passing nullptr/empty triggers auto-detection of the
    //                    Python 3.12 installation: HOOPS_AI_PYTHON_HOME env var, then the
    //                    PEP 514 registry (HKCU/HKLM Software\Python\PythonCore\3.12\InstallPath).
    //                    A candidate is used only if it actually contains python312.dll.
    //                    Does NOT depend on pyvenv.cfg. A non-empty value here overrides detection.
    // licenseKey:        HOOPS AI license key. The bridge does NOT read any environment
    //                    variable for this; supplying the key is entirely the caller's
    //                    responsibility (a client typically compiles it into its own binary
    //                    at build time). Passing nullptr/empty skips set_license altogether,
    //                    which leaves the license unset and later calls failing validation.
    HOOPSAI_API bool HoopsAI_Initialize(const char* venvSitePackages,
                                         const char* pythonHome,
                                         const char* licenseKey,
                                         char* outErrorMsg, int errorMsgSize);

    // Always call this just before the application exits. Releases the Python interpreter.
    HOOPSAI_API void HoopsAI_Shutdown();

    // --- MFR (Manufacturing Feature Recognition): per-face feature classification ---

    // Load an MFR model (for example, ts3d_162k_mfr.ckpt), or switch to a different one.
    // Calling again with the same checkpointPath is a no-op that returns true (the loaded
    // model is reused, avoiding a costly rebuild). Passing a different checkpointPath loads
    // the new model and replaces the current one. The swap happens only after the new model
    // loads successfully; if the load fails, false is returned and the previously loaded
    // model is left intact.
    HOOPSAI_API bool HoopsAI_LoadMFRModel(const char* checkpointPath,
                                           char* outErrorMsg, int errorMsgSize);

    // Run MFR inference on one CAD file (requires HoopsAI_LoadMFRModel to have been called).
    // outLabels: int array allocated by the caller (per-face label IDs, 0 = no feature)
    HOOPSAI_API bool HoopsAI_RunMFRInference(const char* cadFilePath,
                                              int* outLabels, int maxLabels,
                                              int* outLabelCount,
                                              char* outErrorMsg, int errorMsgSize);

    // --- Shape Embeddings: shape similarity ---

    // Load an embeddings model (for example, ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt),
    // or switch to a different one. Calling again with the same checkpointPath is a no-op that
    // returns true. Passing a different checkpointPath loads the new model and replaces the
    // current one (the swap happens only on success; on failure the previous model is kept).
    // Because vectors from different models are not comparable, actually loading a different
    // model closes the current similar-parts index (its in-memory handle is released only;
    // the index files on disk are NOT deleted). The correct order is therefore: load the
    // embeddings model first, then call HoopsAI_OpenIndex to open (or create) an index for
    // that model. A same-path no-op reload does not close the current index.
    HOOPSAI_API bool HoopsAI_LoadEmbeddingsModel(const char* checkpointPath,
                                                  char* outErrorMsg, int errorMsgSize);

    // Compute the shape embedding vector for one CAD file (requires HoopsAI_LoadEmbeddingsModel to have been called).
    // outVector: float array allocated by the caller. Because the dimension count (dim) is not known in advance,
    //            up to maxLen values are written, and the actual dimension count is returned in outDim (typically 2048).
    // [Multi-body] embed_shape returns one vector per body; this call L2-normalizes each, averages
    //            them, and L2-normalizes again into a SINGLE vector per file, for a simple 1-to-1
    //            interface. This differs from index registration (HoopsAI_AddCADToIndex /
    //            HoopsAI_AddCADFolderToIndex), which keeps one row PER body. The averaged vector is
    //            fine for single-body parts but loses assembly composition; for rigorous multi-body
    //            comparison use HoopsAI_SearchSimilarAssembly on a per-body index instead.
    HOOPSAI_API bool HoopsAI_ComputeEmbedding(const char* cadFilePath,
                                               float* outVector, int maxLen, int* outDim,
                                               char* outErrorMsg, int errorMsgSize);

    // Compute the shape similarity of two CAD files (internally calls ComputeEmbedding twice and
    // compares them using L2 normalization + cosine similarity, like compare_embeddings in the Web API version).
    // outSimilarity: -1.0 to 1.0 (closer to 1.0 means more similar)
    HOOPSAI_API bool HoopsAI_CompareEmbeddings(const char* cadFilePath1,
                                                const char* cadFilePath2,
                                                float* outSimilarity,
                                                char* outErrorMsg, int errorMsgSize);

    // --- Similar Parts Index: FAISS-backed index for similar-part search ---
    // The client decides where each index lives and what it is called: there is no fixed
    // name and the bridge holds no default path. An index is identified by a .faiss FILE
    // the client opens or creates; its parent folder name is free. Multiple indexes can
    // exist on disk, and HoopsAI_OpenIndex opens one and makes it the current index (opening
    // a different .faiss switches which one is current).
    // HoopsAI_Initialize() does NOT prepare any index; there is no current index
    // until HoopsAI_OpenIndex is called. Because the index dimension is derived from the
    // loaded embeddings model, load the embeddings model before creating a new index.
    //
    // [On-disk layout] An index is a .faiss FILE plus two siblings created next to it.
    // Given the client opens/creates <dir>/<name>.faiss (base = <dir>/<name>):
    //     <dir>/
    //       <name>.faiss       FAISS vectors (the file the client selected)
    //       <name>.meta        per-record metadata
    //       <name>/            per-index folder (named after the index);
    //                          holds hoops_ai-rendered per-part white-background thumbnails,
    //                          laid out as <name>/<parent>/<stem>_white.png plus a sibling
    //                          <stem>.scs (created during embedding on first add)
    // The bridge passes <dir>/<name> (no extension) as the base path to
    // FaissVectorStore.save/load and detects an existing index by <dir>/<name>.faiss.
    // Saves are atomic (write to a temp base, then rename over <name>.faiss / <name>.meta).
    // [Thumbnails] HoopsAI_AddCADToIndex / HoopsAI_AddCADFolderToIndex ask hoops_ai to render
    // each part while it is embedded, by passing embed_shape_batch the specification
    //     {"generate_images": True, "images_out_dir": <dir>/<name>}.
    // hoops_ai writes the PNG (and a .scs) as <parent folder of the CAD file>/<file stem>_white.png
    // under that folder; the exact relative path it returns is stored in the record metadata
    // ("thumbnail"). Resolve a part's PNG with HoopsAI_GetPartThumbnailPath (it prefers that
    // stored path, then falls back to the deterministic rule and to the legacy "<stem>.png"
    // name so older / WebAPI indexes still resolve).
    // To delete an index, close it with HoopsAI_CloseIndex first, then remove the
    // <name>.faiss / <name>.meta files and the <name>/ folder yourself (the bridge never
    // deletes index files).
    // [Note] As with this bridge overall, these index functions are also reference PoC
    // implementations and should not be exposed directly as an end-user-facing public API
    // (see the "design note" at the top of this header for details).
    //
    // [Thread-safety] These index functions are thread-safe like every other export (see the
    // note on HoopsAI_Initialize at the top of this header). One index-specific caveat: thumbnail
    // rendering can take several seconds per part, so run add/search off the UI thread.

    // Open an index and make it the current one. Same style as loading a ckpt:
    //   - If the same .faiss file is already current, this is a no-op that returns true.
    //   - Otherwise it loads/creates the index and swaps the current one ONLY on success
    //     (on failure the existing current index is kept).
    // faissFilePath: path to the index .faiss FILE (UTF-8). The file stem is the index name
    //            and its parent folder name is free. The bridge derives base = <dir>/<stem>
    //            and uses <dir>/<stem>.faiss / <stem>.meta / <stem>/ next to it. The client
    //            decides the location and name; the bridge has no default. If the path has no
    //            ".faiss" extension it is treated as the base path as-is.
    // createIfMissing: when true, if <dir>/<stem>.faiss does not exist the parent folder and
    //            the per-index <stem>/ folder are created (recursively) and an empty index is
    //            written. When false, a missing index is an error.
    // [Requirement] Creating a new index requires the current embeddings model to be
    //            loaded (the index dimension is derived from it). If it is not loaded, the
    //            call fails with "load an embeddings model first (the index dimension is
    //            derived from it)" and the current index is left unchanged.
    // [Dimension check] When opening an existing index, if an embeddings model is loaded
    //            and its dimension does not match the index dimension, the call fails and
    //            the current index is left unchanged.
    HOOPSAI_API bool HoopsAI_OpenIndex(const char* faissFilePath, bool createIfMissing,
                                        char* outErrorMsg, int errorMsgSize);

    // Release the current index (does NOT delete files on disk). Returns true even when
    // there is no current index (idempotent).
    // [Important] To delete index files manually, always close the current index with this
    // function first. If you delete the files while the index is still open, the next
    // registration's save will recreate them and your deletion will be undone.
    HOOPSAI_API bool HoopsAI_CloseIndex(char* outErrorMsg, int errorMsgSize);

    // Compute the shape embedding for one CAD file and register it in the CURRENT index
    // (requires HoopsAI_LoadEmbeddingsModel and HoopsAI_OpenIndex to have been called).
    // If there is no current index, fails with "no current index: call HoopsAI_OpenIndex first".
    // partId: ID string to register (if nullptr or empty, cadFilePath itself is used as the ID).
    //         If the same ID already exists, it is deleted first and then re-registered
    //         (to prevent duplicates on the FAISS side).
    // outIndexCount: total number of entries in the index after registration.
    // The index is automatically saved (atomic upsert + save) after every successful
    // registration.
    // [Implementation] This delegates to the same embed_shape_batch path as
    // HoopsAI_AddCADFolderToIndex (with a one-element list), so the embedding and the thumbnail
    // are produced in a SINGLE CAD load. embed_shape_batch returns one row per body, so a
    // multi-body file registers several per-body vectors sharing the file path as id.
    // [Thumbnail] hoops_ai renders a white-background thumbnail during embedding into
    // <dir>/<name>/<parent>/<stem>_white.png (see the layout note above); the relative path is
    // stored in the record metadata. Resolve it with HoopsAI_GetPartThumbnailPath.
    HOOPSAI_API bool HoopsAI_AddCADToIndex(const char* cadFilePath,
                                            const char* partId,
                                            int* outIndexCount,
                                            char* outErrorMsg, int errorMsgSize);

    // Batch variant of HoopsAI_AddCADToIndex for adding a whole folder of CAD files at once
    // (requires HoopsAI_LoadEmbeddingsModel and HoopsAI_OpenIndex to have been called).
    // Both entry points share one implementation: this calls the bulk API
    // embedder.embed_shape_batch(cad_path_list, ..., specifications={"generate_images": True,
    // "images_out_dir": <dir>/<name>}) once, builds one record per returned row, upserts them
    // all in a single call, and saves the index exactly ONCE while hoops_ai renders each
    // thumbnail in the same CAD load. That avoids the O(N^2) disk I/O of re-saving the whole
    // FAISS index per file and the former separate rendering pass. Prefer this over a
    // HoopsAI_AddCADToIndex loop when registering many files.
    // cadFilePaths / fileCount: array of UTF-8 CAD file paths and its length. Each file is
    //            registered under its own path as the id (same default as HoopsAI_AddCADToIndex
    //            with partId == nullptr). Re-registering an existing id replaces it (delete +
    //            upsert), so a duplicate is never created.
    // numWorkers: number of parallel worker processes for embed_shape_batch. <= 0 lets the
    //            BRIDGE choose a bounded worker count (1 for small batches, otherwise the smaller
    //            of a cap, half the logical cores, and available-RAM / per-worker-model-footprint)
    //            and is the recommended value. It deliberately does not delegate to hoops_ai's own
    //            all-logical-core auto-detect: because every spawned worker reloads the ~2 GB
    //            embedding model, all-core parallelism oversubscribes RAM and, past ~8 workers,
    //            both slows the batch and drops bodies from the index via per-item timeouts
    //            (benchmarks peaked at 8 workers; all-core ran slower than sequential). The caps
    //            are tunable at runtime via the HOOPS_AI_MAX_WORKERS, HOOPS_AI_MODEL_FOOTPRINT_MB
    //            and HOOPS_AI_MIN_FILES_PARALLEL environment variables. HoopsAI_Initialize repoints
    //            Python's multiprocessing at a real python.exe so these spawn-based workers launch
    //            a clean interpreter instead of relaunching the host process; if that one-time
    //            setup failed, the bridge safely clamps to 1 worker. Pass a specific positive value
    //            only to override the bridge's choice.
    // timeLimitSeconds: per-file embedding time budget passed to embed_shape_batch via the
    //            specifications keys time_limit_overall / time_limit_small / time_limit_medium /
    //            time_limit_large (all set to this value; default 120 s). <= 0 leaves hoops_ai's
    //            default in place. time_limit_overall governs the per-item "Timeout (CUMULATIVE)"
    //            that drops heavy assemblies -- setting only the size buckets leaves it at 120 s,
    //            so all four are set (verified by benchmark; a large value does not abort the
    //            batch). A heavy assembly that exceeds the budget is dropped with a "Timeout" and
    //            reported through outFailedPaths, so a caller can implement a two-pass strategy:
    //            pass 1 with the default budget for the many light files, then pass 2 that re-adds
    //            only the failed paths with a larger timeLimitSeconds. (Benchmark note: dropping
    //            numWorkers to 1 in pass 2 gives NO extra recovery over a modest count like 3 and
    //            is ~2x slower, since a raised time budget alone recovers every time-limited file;
    //            genuine CAD errors remain unrecoverable at any setting.)
    // outAddedCount / outFailedCount: number of distinct files successfully added / that failed.
    // outFailedPaths (optional, may be nullptr): newline-delimited UTF-8 list of the input paths
    //            that failed, NUL-terminated and truncated to outFailedPathsBufSize. It is filled
    //            only when it can be derived unambiguously; outFailedCount is always accurate.
    // outIndexCount: total number of entries in the index after the batch.
    // [Aggregation] embed_shape_batch returns one row per body (each L2-normalized here),
    // matching TechSoft's official "index with embed_shape_batch, query with embed_shape(q)[0]"
    // workflow. Multi-body files therefore register several per-body vectors sharing the file
    // path as id, rather than a single averaged vector.
    // [Thumbnail] hoops_ai renders each part during embedding into
    // <dir>/<name>/<parent>/<stem>_white.png; the relative path is stored in the record
    // metadata and resolved by HoopsAI_GetPartThumbnailPath.
    HOOPSAI_API bool HoopsAI_AddCADFolderToIndex(const char* const* cadFilePaths,
                                                  int fileCount,
                                                  int numWorkers,
                                                  int timeLimitSeconds,
                                                  int* outAddedCount,
                                                  int* outFailedCount,
                                                  char* outFailedPaths, int outFailedPathsBufSize,
                                                  int* outIndexCount,
                                                  char* outErrorMsg, int errorMsgSize);

    // --- Live progress reporting for HoopsAI_AddCADFolderToIndex (optional) ---
    //
    // hoops_ai's embed_shape_batch renders a tqdm progress bar on the parent process while the
    // worker pool computes embeddings. The bridge can forward that live progress to a native
    // caller (e.g. a Qt progress bar) by parsing the tqdm output. Register a callback with
    // HoopsAI_SetProgressCallback BEFORE calling HoopsAI_AddCADFolderToIndex; pass nullptr to
    // stop forwarding. The callback fires from the worker/calling thread (NOT the host GUI
    // thread), possibly many times per second, so it must be cheap and thread-safe -- typically
    // it just marshals the values to the GUI thread (e.g. a queued Qt signal) and returns.
    //
    //   phase:  0 = main worker pool ("Computing embeddings"), 1 = heavy-file fallback
    //           ("Heavy Files (1 worker)", hoops_ai's built-in RAM-driven single-worker retry),
    //           -1 = unknown/other bar.
    //   done / total: items processed so far / total items in the current bar (per phase).
    //   errors: cumulative failure count reported by the bar, or -1 if not present.
    //   heavy:  number of files deferred to the heavy fallback so far, or -1 if not present.
    //   userData: the opaque pointer supplied to HoopsAI_SetProgressCallback.
    //
    // NOTE: the values are parsed from tqdm's human-readable text, so the exact fields
    // (errors=, heavy=) depend on the installed tqdm/hoops_ai versions; done/total/phase are
    // robust, the postfix counters are best-effort.
    typedef void (*HoopsAI_ProgressCallback)(int phase, int done, int total,
                                             int errors, int heavy, void* userData);

    // Register (or clear, with callback == nullptr) the folder-add progress callback. Thread-safe.
    HOOPSAI_API void HoopsAI_SetProgressCallback(HoopsAI_ProgressCallback callback, void* userData);

    // Search the CURRENT index for the topK most similar parts using one CAD file
    // (requires HoopsAI_LoadEmbeddingsModel and HoopsAI_OpenIndex to have been called).
    // If there is no current index, fails with "no current index: call HoopsAI_OpenIndex first".
    // outIds:    character buffer allocated by the caller. Hit IDs are written separated by newlines
    //            (up to outIdsBufSize, always NUL-terminated).
    // outScores: float array allocated by the caller. Writes each hit score (up to maxResults entries).
    // outResultCount: actual number of hits written. If the index is empty, returns 0 and
    //                 completes successfully (not treated as an error; same behavior as Web API search_index).
    // To show a thumbnail for each hit, pass its id to HoopsAI_GetPartThumbnailPath.
    HOOPSAI_API bool HoopsAI_SearchIndex(const char* cadFilePath, int topK,
                                          char* outIds, int outIdsBufSize,
                                          float* outScores, int maxResults,
                                          int* outResultCount,
                                          char* outErrorMsg, int errorMsgSize);

    // Get information about the current index. Can be called even when there is no current
    // index (it does not error; check outHasIndex to tell them apart).
    // outPath:     .faiss file path of the current index (UTF-8, '/'-separated; empty when no
    //              current index). Passing nullptr skips writing the path.
    // outHasIndex: whether a current index exists. When false, outCount / outDim become 0.
    // Returns true even when there is no current index (use outHasIndex to decide).
    HOOPSAI_API bool HoopsAI_GetCurrentIndexInfo(char* outPath, int outPathBufSize,
                                                  bool* outHasIndex,
                                                  int* outCount, int* outDim,
                                                  char* outErrorMsg, int errorMsgSize);

    // Extended statistics for the CURRENT index, distinguishing files from bodies. This is the
    // complete superset of HoopsAI_GetCurrentIndexInfo (it returns the same path/dim plus a full
    // composition breakdown), so new callers can use this single call; HoopsAI_GetCurrentIndexInfo
    // is retained only for backward compatibility. Because the bridge stores one vector per body
    // (all bodies of a file share the file path as id), the "count" from GetCurrentIndexInfo is the
    // number of distinct files, NOT bodies; this call breaks that down.
    //   outPath:            .faiss file path of the current index (UTF-8, '/'-separated; empty when
    //                       no current index). Passing nullptr skips writing the path.
    //   outHasIndex:        whether a current index exists (when false, all counts are 0).
    //   outFileCount:       distinct source files (ids) in the index (== GetCurrentIndexInfo count).
    //   outBodyCount:       total per-body vectors stored (sum of every file's body count).
    //   outAssemblyCount:   files with >= 2 bodies (treated as assemblies).
    //   outSinglePartCount: files with exactly 1 body (single-part files).
    //   outDim:             embedding dimension.
    // Any out-pointer may be nullptr. Returns true even when there is no current index (all
    // outputs 0/empty, outHasIndex=false); check outHasIndex to distinguish that from an empty index.
    HOOPSAI_API bool HoopsAI_GetIndexStats(char* outPath, int outPathBufSize,
                                            bool* outHasIndex,
                                            int* outFileCount, int* outBodyCount,
                                            int* outAssemblyCount, int* outSinglePartCount,
                                            int* outDim,
                                            char* outErrorMsg, int errorMsgSize);

    // Assembly-to-assembly similarity search over the CURRENT index. Where HoopsAI_SearchIndex
    // ranks individual parts, this ranks whole ASSEMBLIES: it groups the corpus's per-body
    // vectors by file, then scores each candidate assembly against the query with an optimal
    // one-to-one (Hungarian) part matching plus an optional TF-IDF rare-part weighting and a
    // bag-of-parts composition blend (mirrors the demo_assembly_to_assembly_retrieval tutorial).
    // The first call on a given index builds rarity weights (a few seconds on a large corpus);
    // the result is cached per index so later calls are fast.
    //   cadFilePath:   the query assembly. If it is already in the index its stored body vectors
    //                  are reused (no model needed); otherwise HoopsAI_LoadEmbeddingsModel must
    //                  have been called so the file can be embedded live.
    //   topK:          maximum assemblies to return.
    //   candidateK:    Stage-1 shortlist size per query body (<= 0 uses a default of 30).
    //   simThresh:     minimum per-part cosine to count a matched pair (<= 0 uses 0.80).
    //   bopWeight:     blend weight of the bag-of-parts composition score vs. the geometric score,
    //                  in [0,1] (< 0 uses 0.30; 0 = geometry only).
    //   coverageMode:  "symmetric" (default; matched / larger side), "containment" (matched /
    //                  query size), or "jaccard". nullptr/empty uses "symmetric".
    //   useIdf:        weight rare parts higher so common hardware does not dominate.
    //   outIds:        newline-delimited assembly ids (file paths), NUL-terminated, up to
    //                  outIdsBufSize.
    //   outScores/outGeomScores/outCoverages: per-result float arrays (each may be nullptr):
    //                  final blended score, geometry-only score, and coverage fraction.
    //   outMatchedParts/outCandidateParts: per-result int arrays (each may be nullptr): number of
    //                  matched part pairs, and the candidate assembly's total part/body count.
    //   maxResults:    capacity of the out arrays.
    //   outResultCount: number of results written.
    // Fails with "no current index: call HoopsAI_OpenIndex first" when no index is open.
    HOOPSAI_API bool HoopsAI_SearchSimilarAssembly(const char* cadFilePath, int topK,
                                                    int candidateK, float simThresh, float bopWeight,
                                                    const char* coverageMode, bool useIdf,
                                                    char* outIds, int outIdsBufSize,
                                                    float* outScores, float* outGeomScores,
                                                    float* outCoverages,
                                                    int* outMatchedParts, int* outCandidateParts,
                                                    int maxResults, int* outResultCount,
                                                    char* outErrorMsg, int errorMsgSize);

    // List the part IDs registered in the CURRENT index (in the store's natural order).
    // For indexes created by this bridge the ID is the CAD file path that was passed to
    // HoopsAI_AddCADToIndex (partId==nullptr defaults to the file path), so a caller can use each
    // ID both to load the file and, via HoopsAI_GetPartThumbnailPath, to resolve its thumbnail.
    // outIds:        newline-delimited UTF-8 IDs (same format as HoopsAI_SearchIndex's outIds),
    //                NUL-terminated. Size the buffer generously; if the joined text does not fit it
    //                is truncated to outIdsBufSize (the final ID may be partial).
    // limit:         maximum number of IDs to write (<= 0 means "all"). Useful to cap a preview
    //                gallery for performance on very large indexes.
    // outResultCount: number of IDs written.
    // Returns true with 0 results when there is no current index or it is empty (not an error),
    // matching HoopsAI_GetCurrentIndexInfo / HoopsAI_SearchIndex behavior.
    HOOPSAI_API bool HoopsAI_ListIndexParts(char* outIds, int outIdsBufSize,
                                             int limit, int* outResultCount,
                                             char* outErrorMsg, int errorMsgSize);

    // Resolve the thumbnail PNG path for a part registered in the CURRENT index.
    // The path is returned as an absolute path with '/' separators, NUL-terminated. The name is
    // resolved against the per-index folder in this order: (1) the relative path stored in the
    // part's metadata "thumbnail" key (what hoops_ai generate_images wrote, e.g.
    // "<parent>/<stem>_white.png"); (2) the deterministic generate_images rule derived from the
    // id; (3) the legacy flat "<stem>.png" name, so indexes created by HOOPS_AI-WebAPI or an
    // older bridge still resolve. The first candidate that exists on disk is returned.
    // outExists: whether the file is actually present on disk. The call still succeeds
    //            (returns true) when the part is registered but its PNG is missing.
    // outPath / outExists may be nullptr (skipped), matching the existing API style.
    // Fails only when there is no current index, partId is null/empty, the part is not
    // registered, or outPath is too small (the required size is included in outErrorMsg).
    // This call touches no CAD data and performs no rendering, so it is cheap enough to call
    // once per search hit.
    HOOPSAI_API bool HoopsAI_GetPartThumbnailPath(const char* partId,
                                                   char* outPath, int outPathBufSize,
                                                   bool* outExists,
                                                   char* outErrorMsg, int errorMsgSize);

    // Overrides the base directory used to resolve part thumbnails (PNGs) for the CURRENT and
    // subsequent lookups. By default the bridge resolves thumbnails under the per-index folder
    // (<indexBase>/, named after the .faiss file). Some indexes ship their images in a separate
    // folder that does not follow that convention (e.g. the tutorial's "images_tmcad"); point
    // this at that folder so HoopsAI_GetPartThumbnailPath / HoopsAI_ListIndexPartsPaged find the
    // "<parent>/<stem>_white.png" (or legacy "<stem>.png") files there.
    //   dir: absolute path to the image root. A null or empty string CLEARS the override,
    //        reverting to the default per-index folder.
    // The override is automatically cleared whenever HoopsAI_OpenIndex opens a (new) index, so a
    // client that wants it to persist must re-apply it after each open. No filesystem validation
    // is performed: a missing folder simply makes thumbnails resolve as "not present".
    HOOPSAI_API bool HoopsAI_SetThumbnailDir(const char* dir,
                                              char* outErrorMsg, int errorMsgSize);

    // Number of body vectors stored for a single part id in the CURRENT index. Because the bridge
    // stores one vector per body and all bodies of a file share the file path as id, this is the
    // part's body/component count: 1 == single-part file, >= 2 == assembly (the same rule used by
    // HoopsAI_GetIndexStats' assembly/single-part breakdown).
    //   partId:        the registered id (CAD file path) to inspect.
    //   outBodyCount:  receives the body count (>= 1 for a registered part). May be nullptr.
    //   outIsAssembly: receives whether the part has >= 2 bodies. May be nullptr.
    // Fails when there is no current index, partId is null/empty, or the part is not registered.
    HOOPSAI_API bool HoopsAI_GetPartBodyCount(const char* partId,
                                               int* outBodyCount, bool* outIsAssembly,
                                               char* outErrorMsg, int errorMsgSize);

    // Paged listing of the CURRENT index that returns, in a SINGLE call, a window of part IDs
    // together with each part's resolved thumbnail path and on-disk presence. This is the
    // large-index-friendly counterpart to calling HoopsAI_ListIndexParts followed by one
    // HoopsAI_GetPartThumbnailPath per id: it resolves get_ids() once, builds the id->thumbnail
    // metadata map in one pass (O(n) instead of the per-id O(n) membership+metadata scan), and so
    // avoids the O(n^2) blow-up when enumerating tens of thousands of parts.
    //
    // offset:          0-based start index into the index's natural id order (clamped to [0,total]).
    // count:           number of parts to return starting at offset (<= 0 means "to the end").
    // outIds:          newline-delimited UTF-8 IDs for the window, NUL-terminated (see truncation
    //                  note below). Split by '\n'; there are exactly *outResultCount ids.
    // outThumbs:       newline-delimited absolute thumbnail paths ('/'-separated), positionally
    //                  parallel to outIds. A part whose thumbnail could not be derived yields an
    //                  empty segment; callers should zip by index (a trailing empty segment may be
    //                  dropped, so treat a missing i-th thumb as empty) and rely on outExists.
    // outKinds:        newline-delimited "part"/"assembly" strings, positionally parallel to
    //                  outIds (empty segment when a record has no stored "kind"). Read from record
    //                  metadata in the same pass as outThumbs, so it adds no measurable cost.
    // outExists:       optional byte array (>= maxResults) receiving 1/0 for each returned part in
    //                  order (whether the thumbnail PNG is present on disk). May be nullptr.
    // maxResults:      capacity of outExists and the hard cap on the number of parts returned.
    // outResultCount:  number of parts actually written (<= maxResults and <= count).
    // outTotalCount:   total number of parts in the current index (for paging math). May be nullptr.
    // Returns true with 0 results / 0 total when there is no current index or it is empty. Buffers
    // that are too small are truncated (size them per page, e.g. a few MB for a whole-index fetch).
    HOOPSAI_API bool HoopsAI_ListIndexPartsPaged(int offset, int count,
                                                  char* outIds, int outIdsBufSize,
                                                  char* outThumbs, int outThumbsBufSize,
                                                  char* outKinds, int outKindsBufSize,
                                                  unsigned char* outExists, int maxResults,
                                                  int* outResultCount, int* outTotalCount,
                                                  char* outErrorMsg, int errorMsgSize);

    // Compute a low-dimensional "shape map" of the CURRENT index for cluster visualization.
    // Reads the stored per-body vectors (no CAD re-embedding), aggregates them to ONE vector per
    // file id (mean of the id's body rows, L2-renormalized), projects the aggregated vectors to
    // 2D/3D via PCA (numpy SVD), and assigns each point a cluster id via FAISS k-means. This is
    // O(N*dim) memory + a few seconds of compute even for indexes with tens of thousands of files,
    // unlike a full N*N similarity/MDS pass.
    //
    //   nClusters:      requested number of clusters (<= 0 => auto: ~min(max(2, N/50), 24)).
    //   dims:           projection dimensionality, 2 or 3 (anything >= 3 => 3; else 2). The Z of a
    //                   2D projection is returned as 0. Coordinates are scaled into a stable
    //                   [-1, 1] cube so the caller can render them at a fixed view scale.
    //   filterIds:      optional newline-delimited UTF-8 list of file ids to restrict the map to
    //                   (the projection and clustering are computed on this subset alone). Pass
    //                   nullptr or "" to map the whole index. Ids not present in the index are
    //                   ignored; if none match, the call succeeds with zero points.
    //   outIds:         newline-delimited UTF-8 file ids (one per point), NUL-terminated. Size the
    //                   buffer generously (a few MB for a whole-index fetch); truncation drops/
    //                   splits the last id, matching HoopsAI_ListIndexParts.
    //   outCoords:      float array of maxPoints*3, row-major [x0,y0,z0, x1,y1,z1, ...]. May be
    //                   nullptr to skip.
    //   outClusterIds:  int array of maxPoints, one cluster id per point (0-based). May be nullptr.
    //   maxPoints:      capacity of outCoords (as maxPoints*3 floats) and outClusterIds, and the
    //                   hard cap on the number of points written. Size it from the index file
    //                   count (HoopsAI_GetIndexStats.outFileCount).
    //   outPointCount:  number of points actually written (<= maxPoints).
    //   outClusterCount: number of clusters actually produced. May be nullptr.
    // Fails with "no current index: call HoopsAI_OpenIndex first" when no index is open.
    HOOPSAI_API bool HoopsAI_ComputeIndexShapeMap(int nClusters, int dims,
                                                   char const* filterIds,
                                                   char* outIds, int outIdsBufSize,
                                                   float* outCoords, int* outClusterIds,
                                                   int maxPoints, int* outPointCount,
                                                   int* outClusterCount,
                                                   char* outErrorMsg, int errorMsgSize);
}
