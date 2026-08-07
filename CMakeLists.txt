# CMakeLists.txt
# Builds only the hoops_ai_bridge shared library (Windows: hoops_ai_bridge.dll,
# Linux: libhoops_ai_bridge.so). A configuration that can be built on both
# Windows (Visual Studio 2022) and GNU/Linux (gcc/clang), since hoops_ai
# itself supports Windows (x86_64) and Linux (x86_64, arm64-v8a).
#
# The samples/test_client.cpp console client is a separate, standalone CMake
# project (see samples/CMakeLists.txt) and is not part of this build. Build
# the bridge here first, then build the sample from samples/ if needed.
#
# Build example (Windows, from a Developer Command Prompt):
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
#   -> hoops_ai_bridge.dll is produced directly under bin\win64\ and its
#      import library under lib\win64\ (hoops_ai_bridge.lib), with no
#      per-configuration "Release\" subfolder.
#      find_package(Python3 3.12 EXACT ...) below auto-detects a registered
#      Python 3.12 (via the PEP 514 registry), so -DPython3_ROOT_DIR is
#      normally unnecessary. Pass it only if detection fails or picks the
#      wrong interpreter, e.g.:
#        -DPython3_ROOT_DIR="C:\Users\toshi\AppData\Local\Programs\Python\Python312"
#
# Build example (Ubuntu 24.04, etc., assuming a Python 3.12 venv for hoops_ai exists):
#   cmake -S . -B build          # activate the venv first, or pass -DPython3_ROOT_DIR=/path/to/.venv
#   cmake --build build
#   -> libhoops_ai_bridge.so is produced under bin/linux64/.
#
cmake_minimum_required(VERSION 3.20)
project(hoops_ai_bridge_poc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    # The source files are UTF-8 (no BOM) and contain non-ASCII comments,
    # so explicitly tell MSVC to use UTF-8 as the source/execution
    # character set.
    # (Without this, the system ANSI code page (e.g. 932) would be used to
    #  interpret the source, and multi-byte characters could break comment
    #  terminators or tokens, causing syntax errors.)
    add_compile_options(/utf-8)
else()
    # gcc/clang on Linux handle UTF-8 source by default, so no extra
    # configuration is needed there.
    # Only make exported symbols default-visible and hide everything else
    # (paired with __attribute__((visibility("default"))) in hoops_ai_bridge.h).
    add_compile_options(-fvisibility=hidden)
endif()

find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Development)

# Output directory layout mirrors the HOOPS Exchange SDK convention
# (bin/<platform>), but without a toolset suffix (e.g. win64_v142) since the
# published interface of this bridge is a plain C ABI (extern "C") that does
# not depend on the MSVC toolset version.
if (WIN32)
    set(BIN_PLATFORM_DIR win64)
else()
    set(BIN_PLATFORM_DIR linux64)
endif()
set(BRIDGE_BIN_DIR ${CMAKE_SOURCE_DIR}/bin/${BIN_PLATFORM_DIR})
# The import library produced alongside hoops_ai_bridge.dll on Windows
# (hoops_ai_bridge.lib, needed by callers such as a Qt project, or the
# samples/test_client sample, to link against) is kept separate from the
# runtime bin/ folder and tracked in git.
set(BRIDGE_LIB_DIR ${CMAKE_SOURCE_DIR}/lib/${BIN_PLATFORM_DIR})

# --- The shared library called from the native app, or from a Linux console ---
#     Windows: hoops_ai_bridge.dll  /  Linux: libhoops_ai_bridge.so
add_library(hoops_ai_bridge SHARED
    src/hoops_ai_bridge.cpp
    include/hoops_ai_bridge.h
)
target_link_libraries(hoops_ai_bridge PRIVATE Python3::Python)
target_include_directories(hoops_ai_bridge PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(hoops_ai_bridge PRIVATE ${Python3_INCLUDE_DIRS})

# Place the runtime binary directly under bin/<platform>/, not build/ or a
# per-configuration subfolder such as Release/, and its Windows import
# library under lib/<platform>/ so external projects (e.g. a Qt project, or
# samples/test_client) have a stable, git-tracked path to link against.
set_target_properties(hoops_ai_bridge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BRIDGE_BIN_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${BRIDGE_BIN_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${BRIDGE_LIB_DIR})
if (CMAKE_CONFIGURATION_TYPES)
    # Multi-configuration generators (e.g. Visual Studio) otherwise append a
    # per-configuration subfolder (Release/, Debug/, ...) to the output
    # directory; override that per-config property too so everything lands
    # directly in bin|lib/<platform>/.
    foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${cfg} CFG)
        set_target_properties(hoops_ai_bridge PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${CFG} ${BRIDGE_BIN_DIR}
            LIBRARY_OUTPUT_DIRECTORY_${CFG} ${BRIDGE_BIN_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_${CFG} ${BRIDGE_LIB_DIR})
    endforeach()
endif()
