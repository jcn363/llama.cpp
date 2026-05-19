cmake_minimum_required(VERSION 3.14...3.28)
project("ggml" C CXX ASM)

set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 12)
set(GGML_VERSION_PATCH 0)
set(GGML_VERSION_BASE "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)
if(GIT_EXE)
    execute_process(COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GGML_BUILD_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(COMMAND ${GIT_EXE} diff-index --quiet HEAD -- .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GGML_GIT_DIRTY
        ERROR_QUIET
    )
endif()

set(GGML_VERSION "${GGML_VERSION_BASE}")

if(NOT GGML_BUILD_COMMIT)
    set(GGML_BUILD_COMMIT "unknown")
endif()

if(DEFINED GGML_GIT_DIRTY AND GGML_GIT_DIRTY EQUAL 1)
    set(GGML_BUILD_COMMIT "${GGML_BUILD_COMMIT}-dirty")
endif()

include(CheckIncludeFileCXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
else()
    set(GGML_STANDALONE OFF)
    if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()

set(BUILD_SHARED_LIBS_DEFAULT ON)

option(BUILD_SHARED_LIBS           "ggml: build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})
option(GGML_BACKEND_DL             "ggml: build backends as dynamic libraries (requires BUILD_SHARED_LIBS)" OFF)
set(GGML_BACKEND_DIR "" CACHE PATH "ggml: directory to load dynamic backends from (requires GGML_BACKEND_DL")

# general
option(GGML_STATIC "ggml: static link libraries"                     OFF)
option(GGML_NATIVE "ggml: optimize the build for the current system" OFF)
option(GGML_LTO    "ggml: enable link time optimization"             OFF)
option(GGML_CCACHE "ggml: use ccache if available"                   ON)

# debug
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY "ggml: enable all compiler warnings in 3rd party libs" OFF)
option(GGML_GPROF                  "ggml: enable gprof"                                   OFF)
option(GGML_FATAL_WARNINGS    "ggml: enable -Werror flag"    OFF)

# sanitizers
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)

# x86_64 instruction set
option(GGML_SSE42            "ggml: enable SSE 4.2"          ON)
option(GGML_AVX              "ggml: enable AVX"              ON)
option(GGML_AVX_VNNI         "ggml: enable AVX-VNNI"         OFF)
option(GGML_AVX2             "ggml: enable AVX2"             OFF)
option(GGML_BMI2             "ggml: enable BMI2"             OFF)
option(GGML_AVX512           "ggml: enable AVX512F"          OFF)
option(GGML_AVX512_VBMI      "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI      "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16      "ggml: enable AVX512-BF16"      OFF)
option(GGML_FMA              "ggml: enable FMA"              OFF)
option(GGML_F16C             "ggml: enable F16C"             OFF)
option(GGML_AMX_TILE         "ggml: enable AMX-TILE"         OFF)
option(GGML_AMX_INT8         "ggml: enable AMX-INT8"         OFF)
option(GGML_AMX_BF16         "ggml: enable AMX-BF16"         OFF)

option(GGML_CPU_HBM          "ggml: use memkind for CPU HBM" OFF)
option(GGML_CPU_REPACK       "ggml: use runtime weight conversion of Q4_0 to Q4_X_X" ON)
option(GGML_CPU_KLEIDIAI     "ggml: use KleidiAI optimized kernels if applicable" OFF)

option(GGML_CPU_ALL_VARIANTS "ggml: build all variants of the CPU backend (requires GGML_BACKEND_DL)" OFF)

# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
option(GGML_CPU                             "ggml: enable CPU backend"                        ON)
option(GGML_SCHED_NO_REALLOC                "ggml: disallow reallocations in ggml-alloc (for debugging)" OFF)

# 3rd party libs / backends
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             ${GGML_LLAMAFILE_DEFAULT})
option(GGML_CUDA                            "ggml: use CUDA"                                  OFF)
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
option(GGML_CUDA_FORCE_CUBLAS               "ggml: always use cuBLAS instead of mmq kernels"  OFF)
set   (GGML_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING "ggml: max. batch size for using peer access")
option(GGML_CUDA_NO_PEER_COPY               "ggml: do not use peer to peer copies"            OFF)
option(GGML_CUDA_NO_VMM                     "ggml: do not try to use CUDA VMM"                OFF)
option(GGML_CUDA_FA                         "ggml: compile ggml FlashAttention CUDA kernels"  ON)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)
option(GGML_CUDA_GRAPHS                     "ggml: use CUDA graphs (llama.cpp only)"          ${GGML_CUDA_GRAPHS_DEFAULT})
option(GGML_CUDA_NCCL                       "ggml: use NVIDIA Collective Comm. Library"       ON)
set   (GGML_CUDA_COMPRESSION_MODE "size" CACHE STRING "ggml: cuda link binary compression mode; requires cuda 12.8+")
set_property(CACHE GGML_CUDA_COMPRESSION_MODE PROPERTY STRINGS "none;speed;balance;size")
option(GGML_OPENMP                          "ggml: use OpenMP"                                ON)

# extra artifacts
option(GGML_BUILD_TESTS    "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES "ggml: build examples" ${GGML_STANDALONE})

#
# dependencies
#

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include(GNUInstallDirs)

#
# build the library
#

add_subdirectory(src)

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (GGML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

#
# install
#

include(CMakePackageConfigHelpers)

set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-cpu.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-blas.h
    include/ggml-cuda.h
    include/ggml-opt.h
    include/ggml-rpc.h
    include/ggml-sycl.h
    include/ggml-vulkan.h
    include/ggml-opencl.h
    include/gguf.h)

set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
install(TARGETS ggml LIBRARY PUBLIC_HEADER)
install(TARGETS ggml-base LIBRARY)

if (GGML_STANDALONE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

# Create CMake package

set(variable_set_statements
"
####### Expanded from @GGML_VARIABLES_EXPANED@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run        #######

")

set(GGML_SHARED_LIB ${BUILD_SHARED_LIBS})

get_cmake_property(all_variables VARIABLES)
foreach(variable_name IN LISTS all_variables)
    if(variable_name MATCHES "^GGML_")
        string(REPLACE ";" "\\;"
               variable_value "${${variable_name}}")
        set(variable_set_statements
            "${variable_set_statements}set(${variable_name} \"${variable_value}\")\n")
    endif()
endforeach()

set(GGML_VARIABLES_EXPANDED ${variable_set_statements})

set(GGML_INSTALL_VERSION ${GGML_VERSION})
set(GGML_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(GGML_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(GGML_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggml
    PATH_VARS GGML_INCLUDE_INSTALL_DIR
              GGML_LIB_INSTALL_DIR
              GGML_BIN_INSTALL_DIR)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ggml-version.cmake
    VERSION ${GGML_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

target_compile_definitions(ggml-base PRIVATE
    GGML_VERSION="${GGML_INSTALL_VERSION}"
    GGML_COMMIT="${GGML_BUILD_COMMIT}"
)
message(STATUS "ggml version: ${GGML_INSTALL_VERSION}")
message(STATUS "ggml commit:  ${GGML_BUILD_COMMIT}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/ggml-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggml)
