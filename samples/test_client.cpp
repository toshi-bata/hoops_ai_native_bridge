// test_client.cpp
// A portable C++ console client used to validate behavior in place of the
// "calling application" from a Windows native app or Linux console context.
// It can be built and run on both Windows and Linux and does not depend on
// Windows-specific APIs.
//
// In an actual native application, the mfr subcommand processing is expected
// to be moved into the "Run MFR" menu handler, using outLabels[faceIndex] to
// change the material/color of the corresponding face.
//
// Usage:
//   test_client mfr          <cad_file> <mfr_checkpoint.ckpt>
//   test_client embed        <cad_file> <embeddings_checkpoint.ckpt>
//   test_client compare      <cad_file_1> <cad_file_2> <embeddings_checkpoint.ckpt>
//   test_client index-add    <cad_file> <embeddings_checkpoint.ckpt> [--index <basePath>]
//   test_client index-search <cad_file> <K> <embeddings_checkpoint.ckpt> [--index <basePath>]
//   test_client similar-assembly <cad_file> <K> [<embeddings_checkpoint.ckpt>] [--index <basePath>]
//   test_client index-info   [--index <basePath>]
//   test_client index-close  [--index <basePath>]
//
//   --index <basePath> is the index base path. A trailing ".faiss" is stripped; anything
//   else is used as-is. Given base <dir>/<name>, the bridge maintains <name>.faiss,
//   <name>.meta and a <name>/ folder (holding the rendered thumbnails) next to them.
//   When omitted, this sample defaults to "my_index" next to the executable. NOTE: that
//   default is a *sample-side* convenience only; the bridge itself has no default index
//   path and requires the client to choose one.
//
// Environment variables:
//   HOOPS_AI_HOME           HOOPS AI installation directory (the parent of the
//                           venv and packages/trained_ml_models/). Needed only on a
//                           DEVELOPMENT machine that has HOOPS AI installed; the
//                           site-packages path is derived from it as follows:
//                             Windows: %HOOPS_AI_HOME%\.venv\Lib\site-packages
//                             Linux  : $HOOPS_AI_HOME/.venv/lib/python3.12/site-packages
//                           On a REDISTRIBUTION PACKAGE (HOOPS AI not installed) this is
//                           NOT required: the bundled site-packages (.venv next to bin/)
//                           is auto-detected relative to this executable.
//   HOOPS_AI_PYTHON_HOME    Python installation used as the base for the venv.
//                           Optional on Windows: if unset, the bridge
//                           auto-detects the Python 3.12 install from the PEP
//                           514 registry. Set it explicitly only for unusual
//                           setups where auto-detection cannot find it. On
//                           Linux it is normally unnecessary.
//
// License key (required, embedded into the binary at build time):
//   Copy `include/hoops_license.h.example` to `include/hoops_license.h`, then set
//   `#define HOOPS_LICENSE "your_actual_key"` before building. This sample resolves
//   `hoops_license.h` from the repository `include/` directory (added to the include
//   path by samples/CMakeLists.txt), the same place the bridge header lives.
//   `include/hoops_license.h` is ignored by .gitignore, so it is not committed to the
//   repository.
//   If this file is missing or `HOOPS_LICENSE` is not defined, the build fails
//   (to require a distribution model where partners build it into their own
//   product and ship it without exposing the license key to end users, loading
//   from environment variables is not supported).
#include "hoops_ai_bridge.h"
#include "hoops_license.h"
#ifndef HOOPS_LICENSE
#error "Please set #define HOOPS_LICENSE \"your_actual_key\" in include/hoops_license.h (see include/hoops_license.h.example)"
#endif
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace {

std::string GetEnvOrDefault(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

// Directory containing this executable (portable). Used only to build the sample-side
// default index path; it is NOT how the bridge locates indexes (the bridge has no default).
std::string GetExecutableDir() {
#if defined(_WIN32)
    wchar_t buf[MAX_PATH];
    DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (len == 0 || len == MAX_PATH) return ".";
    std::wstring wpath(buf, len);
    size_t pos = wpath.find_last_of(L"\\/");
    if (pos == std::wstring::npos) return ".";
    std::wstring wdir = wpath.substr(0, pos);
    int size = WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return ".";
    std::string dir(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), -1, dir.data(), size, nullptr, nullptr);
    return dir;
#else
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return ".";
    buf[len] = '\0';
    std::string path(buf);
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
#endif
}

// Sample-side default index directory: <executable dir>/my_index. The bridge has no
// default; a real client must always pass its own index directory to HoopsAI_OpenIndex.
std::string DefaultIndexDir() {
    return GetExecutableDir() + "/my_index";
}

// Resolve the venv site-packages directory that contains hoops_ai. Two layouts are
// supported, tried in this order:
//   1) Development machine: HOOPS_AI_HOME points at the installed HOOPS AI, whose venv holds
//      hoops_ai (Windows: <home>\.venv\Lib\site-packages, Linux:
//      <home>/.venv/lib/python3.12/site-packages).
//   2) Redistribution package: HOOPS AI is NOT installed, but the required site-packages was
//      bundled into the package as .venv next to bin/. Since this executable lives in
//      <pkg>/bin/, the bundled site-packages is <exeDir>/../.venv/... — so it resolves
//      automatically and HOOPS_AI_HOME does not need to be set on the target machine.
std::string ResolveSitePackages() {
    namespace fs = std::filesystem;
#ifdef _WIN32
    const fs::path sub = fs::path(".venv") / "Lib" / "site-packages";
#else
    const fs::path sub = fs::path(".venv") / "lib" / "python3.12" / "site-packages";
#endif
    const std::string home = GetEnvOrDefault("HOOPS_AI_HOME", "");
    if (!home.empty())
        return (fs::u8path(home) / sub).u8string();

    // Bundled-package fallback: <exeDir>/../.venv/...
    std::error_code ec;
    const fs::path relative = fs::u8path(GetExecutableDir()) / ".." / sub;
    const fs::path canonical = fs::weakly_canonical(relative, ec);
    return (ec ? relative : canonical).u8string();
}

bool InitializeFromEnv(char* errBuf, int errBufSize) {
    // On a redistribution package the site-packages is auto-detected relative to this
    // executable (see ResolveSitePackages); HOOPS_AI_HOME is only needed on a development
    // machine that has HOOPS AI installed elsewhere.
    const std::string sitePackages = ResolveSitePackages();
    // HOOPS_AI_PYTHON_HOME is optional: when empty we pass nullptr and let the
    // bridge auto-detect the Python 3.12 install from the registry (PEP 514) on
    // Windows. Set it only for unusual setups where detection fails.
    const std::string pythonHome   = GetEnvOrDefault("HOOPS_AI_PYTHON_HOME", "");
    // Always use the license key embedded into hoops_license.h at build time.
    const std::string license = HOOPS_LICENSE;

    std::error_code ec;
    if (sitePackages.empty() || !std::filesystem::exists(std::filesystem::u8path(sitePackages), ec)) {
        std::cerr << "Warning: site-packages not found at '" << sitePackages << "'.\n"
                     "  On a redistribution package, run this exe from inside the package so "
                     "the bundled .venv (next to bin/) is found.\n"
                     "  On a development machine, set HOOPS_AI_HOME to your HOOPS AI install.\n";
    }

    return HoopsAI_Initialize(sitePackages.empty() ? nullptr : sitePackages.c_str(),
                               pythonHome.empty() ? nullptr : pythonHome.c_str(),
                               license.empty() ? nullptr : license.c_str(),
                               errBuf, errBufSize);
}

int RunMFR(const std::string& cadFile, const std::string& checkpoint) {
    char errBuf[8192] = {0};
    std::cout << "[1/3] HoopsAI_Initialize ...\n";
    if (!InitializeFromEnv(errBuf, sizeof(errBuf))) {
        std::cerr << "Initialize failed: " << errBuf << "\n";
        return 1;
    }

    std::cout << "[2/3] HoopsAI_LoadMFRModel " << checkpoint << " ...\n";
    if (!HoopsAI_LoadMFRModel(checkpoint.c_str(), errBuf, sizeof(errBuf))) {
        std::cerr << "LoadMFRModel failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[3/3] HoopsAI_RunMFRInference on " << cadFile << " ...\n";
    std::vector<int> labels(100000, 0);
    int labelCount = 0;
    bool ok = HoopsAI_RunMFRInference(cadFile.c_str(),
                                       labels.data(), static_cast<int>(labels.size()),
                                       &labelCount, errBuf, sizeof(errBuf));
    if (!ok) {
        std::cerr << "RunMFRInference failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "Inference done. faces=" << labelCount << "\n";
    // In a native application, call HC_Set_Color or the A3D-side color API
    // here based on the label assigned to each face.
    for (int i = 0; i < labelCount && i < 20; ++i) {
        std::cout << "  face[" << i << "] label=" << labels[i] << "\n";
    }
    if (labelCount > 20) std::cout << "  ... (" << (labelCount - 20) << " more)\n";

    HoopsAI_Shutdown();
    return 0;
}

int RunEmbed(const std::string& cadFile, const std::string& checkpoint) {
    char errBuf[8192] = {0};
    std::cout << "[1/3] HoopsAI_Initialize ...\n";
    if (!InitializeFromEnv(errBuf, sizeof(errBuf))) {
        std::cerr << "Initialize failed: " << errBuf << "\n";
        return 1;
    }

    std::cout << "[2/3] HoopsAI_LoadEmbeddingsModel " << checkpoint << " ...\n";
    if (!HoopsAI_LoadEmbeddingsModel(checkpoint.c_str(), errBuf, sizeof(errBuf))) {
        std::cerr << "LoadEmbeddingsModel failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[3/3] HoopsAI_ComputeEmbedding on " << cadFile << " ...\n";
    std::vector<float> vec(8192, 0.0f);
    int dim = 0;
    if (!HoopsAI_ComputeEmbedding(cadFile.c_str(), vec.data(), static_cast<int>(vec.size()), &dim,
                                   errBuf, sizeof(errBuf))) {
        std::cerr << "ComputeEmbedding failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "Embedding dim=" << dim << "\n  [";
    for (int i = 0; i < dim && i < 8; ++i) std::cout << vec[i] << (i + 1 < dim && i + 1 < 8 ? ", " : "");
    std::cout << (dim > 8 ? ", ...]\n" : "]\n");

    HoopsAI_Shutdown();
    return 0;
}

int RunCompare(const std::string& cadFile1, const std::string& cadFile2, const std::string& checkpoint) {
    char errBuf[8192] = {0};
    std::cout << "[1/4] HoopsAI_Initialize ...\n";
    if (!InitializeFromEnv(errBuf, sizeof(errBuf))) {
        std::cerr << "Initialize failed: " << errBuf << "\n";
        return 1;
    }

    std::cout << "[2/4] HoopsAI_LoadEmbeddingsModel " << checkpoint << " ...\n";
    if (!HoopsAI_LoadEmbeddingsModel(checkpoint.c_str(), errBuf, sizeof(errBuf))) {
        std::cerr << "LoadEmbeddingsModel failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[3/4] HoopsAI_CompareEmbeddings\n  A: " << cadFile1 << "\n  B: " << cadFile2 << "\n";
    float similarity = 0.0f;
    if (!HoopsAI_CompareEmbeddings(cadFile1.c_str(), cadFile2.c_str(), &similarity,
                                    errBuf, sizeof(errBuf))) {
        std::cerr << "CompareEmbeddings failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[4/4] cosine similarity = " << similarity << "\n";

    HoopsAI_Shutdown();
    return 0;
}

int RunIndexAdd(const std::string& cadFile, const std::string& checkpoint,
                const std::string& indexPath) {
    char errBuf[8192] = {0};
    std::cout << "[1/5] HoopsAI_Initialize ...\n";
    if (!InitializeFromEnv(errBuf, sizeof(errBuf))) {
        std::cerr << "Initialize failed: " << errBuf << "\n";
        return 1;
    }

    // Load the embeddings model BEFORE opening the index: a new index derives its
    // dimension from the loaded model.
    std::cout << "[2/5] HoopsAI_LoadEmbeddingsModel " << checkpoint << " ...\n";
    if (!HoopsAI_LoadEmbeddingsModel(checkpoint.c_str(), errBuf, sizeof(errBuf))) {
        std::cerr << "LoadEmbeddingsModel failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[3/5] HoopsAI_OpenIndex " << indexPath << " (createIfMissing=true) ...\n";
    if (!HoopsAI_OpenIndex(indexPath.c_str(), /*createIfMissing=*/true, errBuf, sizeof(errBuf))) {
        std::cerr << "OpenIndex failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[4/5] HoopsAI_AddCADToIndex " << cadFile << " ...\n";
    int indexCount = 0;
    // partId is nullptr (use cadFilePath itself as the ID)
    if (!HoopsAI_AddCADToIndex(cadFile.c_str(), nullptr, &indexCount, errBuf, sizeof(errBuf))) {
        std::cerr << "AddCADToIndex failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }
    // AddCADToIndex may return true yet still leave a warning (e.g. thumbnail generation
    // failed). A non-empty errBuf on success is a warning, not an error.
    if (errBuf[0] != '\0') {
        std::cout << "  " << errBuf << "\n";
    }

    std::cout << "[5/5] registered. index_count=" << indexCount << "\n";

    // Resolve the thumbnail path for the part we just registered (partId defaulted to the
    // CAD file path). This is cheap: it touches no CAD data and does no rendering.
    char thumbBuf[4096] = {0};
    bool thumbExists = false;
    char thumbErr[8192] = {0};
    if (HoopsAI_GetPartThumbnailPath(cadFile.c_str(), thumbBuf, sizeof(thumbBuf),
                                     &thumbExists, thumbErr, sizeof(thumbErr))) {
        std::cout << "  thumbnail=" << thumbBuf
                  << " exists=" << (thumbExists ? "true" : "false") << "\n";
    } else {
        std::cout << "  thumbnail path unavailable: " << thumbErr << "\n";
    }

    HoopsAI_Shutdown();
    return 0;
}

int RunIndexSearch(const std::string& cadFile, int topK, const std::string& checkpoint,
                   const std::string& indexPath) {
    char errBuf[8192] = {0};
    std::cout << "[1/5] HoopsAI_Initialize ...\n";
    if (!InitializeFromEnv(errBuf, sizeof(errBuf))) {
        std::cerr << "Initialize failed: " << errBuf << "\n";
        return 1;
    }

    std::cout << "[2/5] HoopsAI_LoadEmbeddingsModel " << checkpoint << " ...\n";
    if (!HoopsAI_LoadEmbeddingsModel(checkpoint.c_str(), errBuf, sizeof(errBuf))) {
        std::cerr << "LoadEmbeddingsModel failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[3/5] HoopsAI_OpenIndex " << indexPath << " (createIfMissing=true) ...\n";
    if (!HoopsAI_OpenIndex(indexPath.c_str(), /*createIfMissing=*/true, errBuf, sizeof(errBuf))) {
        std::cerr << "OpenIndex failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[4/5] HoopsAI_SearchIndex on " << cadFile << " (topK=" << topK << ") ...\n";
    std::vector<char> idsBuf(65536, '\0');
    std::vector<float> scores(1024, 0.0f);
    int resultCount = 0;
    if (!HoopsAI_SearchIndex(cadFile.c_str(), topK,
                              idsBuf.data(), static_cast<int>(idsBuf.size()),
                              scores.data(), static_cast<int>(scores.size()),
                              &resultCount, errBuf, sizeof(errBuf))) {
        std::cerr << "SearchIndex failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[5/5] hits=" << resultCount << "\n";
    // outIds is returned as a single newline-delimited string, so split it by
    // line and display each entry with its score and resolved thumbnail path.
    std::string ids(idsBuf.data());
    std::istringstream idsStream(ids);
    std::string line;
    int i = 0;
    while (std::getline(idsStream, line) && i < resultCount) {
        char thumbBuf[4096] = {0};
        bool thumbExists = false;
        char thumbErr[8192] = {0};
        std::string thumbCol = "(unavailable)";
        if (HoopsAI_GetPartThumbnailPath(line.c_str(), thumbBuf, sizeof(thumbBuf),
                                         &thumbExists, thumbErr, sizeof(thumbErr))) {
            thumbCol = thumbBuf;
        }
        std::cout << "  [" << i << "] id=" << line
                  << " score=" << scores[i]
                  << " thumbnail=" << thumbCol
                  << " exists=" << (thumbExists ? "true" : "false") << "\n";
        ++i;
    }

    HoopsAI_Shutdown();
    return 0;
}

int RunIndexInfo(const std::string& indexPath) {
    char errBuf[8192] = {0};
    std::cout << "[1/3] HoopsAI_Initialize ...\n";
    if (!InitializeFromEnv(errBuf, sizeof(errBuf))) {
        std::cerr << "Initialize failed: " << errBuf << "\n";
        return 1;
    }

    // index-info only inspects an EXISTING index: no embeddings model is needed, and it
    // must NOT create a new one. Open with createIfMissing=false (the dimension comes from
    // the file, and the dimension check is skipped when no model is loaded); a missing
    // index is an error.
    std::cout << "[2/3] HoopsAI_OpenIndex " << indexPath << " (createIfMissing=false) ...\n";
    if (!HoopsAI_OpenIndex(indexPath.c_str(), /*createIfMissing=*/false, errBuf, sizeof(errBuf))) {
        std::cerr << "OpenIndex failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    // GetIndexStats is the complete superset of GetCurrentIndexInfo (same path/dim plus the
    // files/bodies/assemblies breakdown), so a single call returns everything.
    std::cout << "[3/3] HoopsAI_GetIndexStats ...\n";
    char pathBuf[4096] = {0};
    bool hasIndex = false;
    int files = 0, bodies = 0, assemblies = 0, single = 0, dim = 0;
    if (!HoopsAI_GetIndexStats(pathBuf, sizeof(pathBuf), &hasIndex,
                               &files, &bodies, &assemblies, &single, &dim,
                               errBuf, sizeof(errBuf))) {
        std::cerr << "GetIndexStats failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    if (hasIndex) {
        std::cout << "path=" << pathBuf << " dim=" << dim << "\n"
                  << "  files=" << files
                  << " bodies=" << bodies
                  << " assemblies=" << assemblies
                  << " single_part_files=" << single << "\n";
    } else {
        std::cout << "no current index\n";
    }

    HoopsAI_Shutdown();
    return 0;
}

// Assembly-to-assembly similarity search. The embeddings model (checkpoint) is OPTIONAL: it is
// only needed when the query file is not already in the index (an out-of-corpus query must be
// embedded live). When the query is in the index its stored body vectors are reused.
int RunSimilarAssembly(const std::string& cadFile, int topK, const std::string& checkpoint,
                       const std::string& indexPath) {
    char errBuf[8192] = {0};
    std::cout << "[1/4] HoopsAI_Initialize ...\n";
    if (!InitializeFromEnv(errBuf, sizeof(errBuf))) {
        std::cerr << "Initialize failed: " << errBuf << "\n";
        return 1;
    }

    if (!checkpoint.empty()) {
        std::cout << "[2/4] HoopsAI_LoadEmbeddingsModel " << checkpoint << " ...\n";
        if (!HoopsAI_LoadEmbeddingsModel(checkpoint.c_str(), errBuf, sizeof(errBuf))) {
            std::cerr << "LoadEmbeddingsModel failed: " << errBuf << "\n";
            HoopsAI_Shutdown();
            return 1;
        }
    } else {
        std::cout << "[2/4] (no checkpoint given: in-corpus query, reusing stored vectors)\n";
    }

    std::cout << "[3/4] HoopsAI_OpenIndex " << indexPath << " (createIfMissing=false) ...\n";
    if (!HoopsAI_OpenIndex(indexPath.c_str(), /*createIfMissing=*/false, errBuf, sizeof(errBuf))) {
        std::cerr << "OpenIndex failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "[4/4] HoopsAI_SearchSimilarAssembly on " << cadFile
              << " (topK=" << topK << ") ...\n";
    std::vector<char> idsBuf(65536, '\0');
    std::vector<float> scores(1024, 0.0f), geom(1024, 0.0f), cover(1024, 0.0f);
    std::vector<int> matched(1024, 0), candParts(1024, 0);
    int resultCount = 0;
    if (!HoopsAI_SearchSimilarAssembly(cadFile.c_str(), topK,
                                       /*candidateK=*/0, /*simThresh=*/0.0f, /*bopWeight=*/-1.0f,
                                       /*coverageMode=*/"symmetric", /*useIdf=*/true,
                                       idsBuf.data(), static_cast<int>(idsBuf.size()),
                                       scores.data(), geom.data(), cover.data(),
                                       matched.data(), candParts.data(),
                                       static_cast<int>(scores.size()),
                                       &resultCount, errBuf, sizeof(errBuf))) {
        std::cerr << "SearchSimilarAssembly failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    std::cout << "hits=" << resultCount << "\n";
    std::string ids(idsBuf.data());
    std::istringstream idsStream(ids);
    std::string line;
    int i = 0;
    while (std::getline(idsStream, line) && i < resultCount) {
        std::cout << "  [" << i << "] score=" << scores[i]
                  << " geom=" << geom[i]
                  << " coverage=" << cover[i]
                  << " matched=" << matched[i]
                  << " parts=" << candParts[i]
                  << " assembly=" << line << "\n";
        ++i;
    }

    HoopsAI_Shutdown();
    return 0;
}

int RunIndexClose(const std::string& indexPath) {
    char errBuf[8192] = {0};
    std::cout << "[1/4] HoopsAI_Initialize ...\n";
    if (!InitializeFromEnv(errBuf, sizeof(errBuf))) {
        std::cerr << "Initialize failed: " << errBuf << "\n";
        return 1;
    }

    // Closing does not need the embeddings model. Open the EXISTING index with
    // createIfMissing=false (the dimension comes from the file, and the dimension
    // check is skipped when no model is loaded), then close it.
    std::cout << "[2/4] HoopsAI_OpenIndex " << indexPath << " (createIfMissing=false) ...\n";
    if (!HoopsAI_OpenIndex(indexPath.c_str(), /*createIfMissing=*/false, errBuf, sizeof(errBuf))) {
        // No existing index: nothing to close. CloseIndex is still idempotent (returns true).
        std::cout << "  no existing index to open (" << errBuf << ")\n";
        HoopsAI_CloseIndex(errBuf, sizeof(errBuf));
        HoopsAI_Shutdown();
        return 0;
    }

    std::cout << "[3/4] HoopsAI_CloseIndex ...\n";
    if (!HoopsAI_CloseIndex(errBuf, sizeof(errBuf))) {
        std::cerr << "CloseIndex failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }

    // Confirm the current index is now gone (outHasIndex=false) yet the files remain.
    char pathBuf[4096] = {0};
    bool hasIndex = true;
    int count = 0, dim = 0;
    if (!HoopsAI_GetCurrentIndexInfo(pathBuf, sizeof(pathBuf), &hasIndex, &count, &dim,
                                      errBuf, sizeof(errBuf))) {
        std::cerr << "GetCurrentIndexInfo failed: " << errBuf << "\n";
        HoopsAI_Shutdown();
        return 1;
    }
    std::cout << "[4/4] closed. has_current_index=" << (hasIndex ? "true" : "false") << "\n";
    std::cout << "  The index files were NOT deleted. To delete this index, remove\n"
              << "  \"" << indexPath << "\".faiss, \"" << indexPath << "\".meta and the\n"
              << "  folder \"" << indexPath << "\" manually now.\n";

    HoopsAI_Shutdown();
    return 0;
}

void PrintUsage(const char* argv0) {
    std::cout << "Usage:\n"
              << "  " << argv0 << " mfr          <cad_file> <mfr_checkpoint.ckpt>\n"
              << "  " << argv0 << " embed        <cad_file> <embeddings_checkpoint.ckpt>\n"
              << "  " << argv0 << " compare      <cad_file_1> <cad_file_2> <embeddings_checkpoint.ckpt>\n"
              << "  " << argv0 << " index-add    <cad_file> <embeddings_checkpoint.ckpt> [--index <basePath>]\n"
              << "  " << argv0 << " index-search <cad_file> <K> <embeddings_checkpoint.ckpt> [--index <basePath>]\n"
              << "  " << argv0 << " similar-assembly <cad_file> <K> [<embeddings_checkpoint.ckpt>] [--index <basePath>]\n"
              << "  " << argv0 << " index-info   [--index <basePath>]\n"
              << "  " << argv0 << " index-close  [--index <basePath>]\n"
              << "\n"
              << "  --index <basePath> : the index base path. A trailing \".faiss\" is stripped;\n"
              << "                       anything else is used as-is. Given base <dir>/<name> the\n"
              << "                       bridge maintains <name>.faiss, <name>.meta and a <name>/\n"
              << "                       folder next to them. Defaults to \"my_index\" next to this\n"
              << "                       executable (a sample-side default; the bridge has none).\n"
              << "  index-add renders a white-background thumbnail per part during embedding, into\n"
              << "    <base>/<parent folder of the CAD file>/<file stem>_white.png;\n"
              << "    index-search prints each hit's resolved thumbnail path.\n"
              << "  similar-assembly ranks whole ASSEMBLIES (not parts) most similar to <cad_file>.\n"
              << "    The checkpoint is optional: needed only when <cad_file> is not already in the\n"
              << "    index (an out-of-corpus query is embedded live); an in-corpus query reuses\n"
              << "    stored vectors. The first search on an index builds rarity weights (a few\n"
              << "    seconds on a large corpus), cached for subsequent searches.\n"
              << "  To delete an index, run 'index-close' first, then delete <name>.faiss,\n"
              << "  <name>.meta and the <name>/ folder manually.\n"
              << "\n"
              << "Environment variables (set before running):\n"
              << "  HOOPS_AI_HOME (required)\n"
              << "  HOOPS_AI_PYTHON_HOME (Windows, optional: auto-detected from the\n"
              << "    PEP 514 registry if unset; set only for unusual setups)\n"
              << "  (license key is embedded at build time via hoops_license.h)\n";
}

// Extract an optional "--index <basePath>" pair from argv (anywhere after the subcommand),
// removing it from args. Returns the value, or the sample-side default when not present.
std::string ExtractIndexPath(std::vector<std::string>& args) {
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--index" && i + 1 < args.size()) {
            std::string path = args[i + 1];
            args.erase(args.begin() + i, args.begin() + i + 2);
            return path;
        }
    }
    return DefaultIndexDir();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];

    // Collect the remaining args and pull out an optional --index <basePath>.
    std::vector<std::string> rest;
    for (int i = 2; i < argc; ++i) rest.push_back(argv[i]);
    const std::string indexPath = ExtractIndexPath(rest);

    if (mode == "mfr" && rest.size() >= 2) {
        return RunMFR(rest[0], rest[1]);
    } else if (mode == "embed" && rest.size() >= 2) {
        return RunEmbed(rest[0], rest[1]);
    } else if (mode == "compare" && rest.size() >= 3) {
        return RunCompare(rest[0], rest[1], rest[2]);
    } else if (mode == "index-add" && rest.size() >= 2) {
        return RunIndexAdd(rest[0], rest[1], indexPath);
    } else if (mode == "index-search" && rest.size() >= 3) {
        return RunIndexSearch(rest[0], std::atoi(rest[1].c_str()), rest[2], indexPath);
    } else if (mode == "similar-assembly" && rest.size() >= 2) {
        const std::string ckpt = rest.size() >= 3 ? rest[2] : std::string();
        return RunSimilarAssembly(rest[0], std::atoi(rest[1].c_str()), ckpt, indexPath);
    } else if (mode == "index-info") {
        return RunIndexInfo(indexPath);
    } else if (mode == "index-close") {
        return RunIndexClose(indexPath);
    }

    PrintUsage(argv[0]);
    return 1;
}
