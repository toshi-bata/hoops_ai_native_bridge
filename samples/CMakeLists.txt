# samples/CMakeLists.txt
# Standalone build for the test_client sample console client.
#
# This is deliberately a separate CMake project from the top-level
# CMakeLists.txt (which only builds hoops_ai_bridge). It does not build
# hoops_ai_bridge itself; it links against the already-built shared library
# / import library under ../bin/<platform>/ and ../lib/<platform>/. Build the
# bridge first (see ../README.md), then build this sample either from here:
#
#   cd samples
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64   (Windows)
#   cmake --build build --config Release
#
#   cd samples
#   cmake -S . -B build                                       (Linux)
#   cmake --build build
#
# or by pointing -S/-B at this folder from anywhere:
#   cmake -S samples -B samples/build ...
#
# test_client.exe / test_client is placed directly under ../bin/<platform>/,
# next to hoops_ai_bridge.dll/.so, so it can resolve it at runtime.
cmake_minimum_required(VERSION 3.20)
project(hoops_ai_bridge_test_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    # See the top-level CMakeLists.txt for why /utf-8 is needed.
    add_compile_options(/utf-8)
endif()

# Root of the hoops_ai_native_bridge repository (parent of this samples/
# folder), used to locate the prebuilt bridge library/headers.
get_filename_component(BRIDGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if (WIN32)
    set(BIN_PLATFORM_DIR win64)
else()
    set(BIN_PLATFORM_DIR linux64)
endif()

set(BRIDGE_BIN_DIR ${BRIDGE_ROOT_DIR}/bin/${BIN_PLATFORM_DIR})
set(BRIDGE_LIB_DIR ${BRIDGE_ROOT_DIR}/lib/${BIN_PLATFORM_DIR})
set(BRIDGE_INCLUDE_DIR ${BRIDGE_ROOT_DIR}/include)

# Locate the prebuilt hoops_ai_bridge library. On Windows this is the
# import library hoops_ai_bridge.lib under lib/win64/ (produced by the
# top-level build); on Linux it is libhoops_ai_bridge.so itself under
# bin/linux64/ (there is no separate import library).
if (WIN32)
    find_library(HOOPS_AI_BRIDGE_LIB
        NAMES hoops_ai_bridge
        PATHS ${BRIDGE_LIB_DIR}
        NO_DEFAULT_PATH)
else()
    find_library(HOOPS_AI_BRIDGE_LIB
        NAMES hoops_ai_bridge
        PATHS ${BRIDGE_BIN_DIR}
        NO_DEFAULT_PATH)
endif()

if (NOT HOOPS_AI_BRIDGE_LIB)
    message(FATAL_ERROR
        "hoops_ai_bridge library not found.\n"
        "Expected under: ${BRIDGE_LIB_DIR} (Windows) or ${BRIDGE_BIN_DIR} (Linux).\n"
        "Build the bridge first from the repository root, e.g.:\n"
        "  cmake -S . -B build -G \"Visual Studio 17 2022\" -A x64 -DPython3_ROOT_DIR=<...>\n"
        "  cmake --build build --config Release")
endif()

add_executable(test_client test_client.cpp)
target_include_directories(test_client PRIVATE ${BRIDGE_INCLUDE_DIR})
target_link_libraries(test_client PRIVATE ${HOOPS_AI_BRIDGE_LIB})

# Place test_client directly under bin/<platform>/, next to
# hoops_ai_bridge.dll/.so, not under samples/build/ or a per-configuration
# subfolder such as Release/.
set_target_properties(test_client PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BRIDGE_BIN_DIR})
if (CMAKE_CONFIGURATION_TYPES)
    foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${cfg} CFG)
        set_target_properties(test_client PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${CFG} ${BRIDGE_BIN_DIR})
    endforeach()
endif()

if (NOT WIN32)
    # Make libhoops_ai_bridge.so, located in the same directory as the
    # executable, discoverable without needing to set LD_LIBRARY_PATH.
    set_target_properties(test_client PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
