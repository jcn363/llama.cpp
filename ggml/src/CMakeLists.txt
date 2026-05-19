include(CheckCXXCompilerFlag)
include("../cmake/common.cmake")

add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (GGML_SANITIZE_THREAD)
    add_compile_options(-fsanitize=thread)
    link_libraries     (-fsanitize=thread)
endif()

if (GGML_SANITIZE_ADDRESS)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    link_libraries     (-fsanitize=address)
endif()

if (GGML_SANITIZE_UNDEFINED)
    add_compile_options(-fsanitize=undefined)
    link_libraries     (-fsanitize=undefined)
endif()

if (GGML_FATAL_WARNINGS)
    list(APPEND C_FLAGS   -Werror)
    list(APPEND CXX_FLAGS -Werror)
endif()

if (GGML_ALL_WARNINGS)
    list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
    list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                              -Werror=implicit-int -Werror=implicit-function-declaration)
    list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

    list(APPEND C_FLAGS   ${WARNING_FLAGS})
    list(APPEND CXX_FLAGS ${WARNING_FLAGS})

    ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

    add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
endif()

if (GGML_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (GGML_CCACHE AND NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(GGML_CCACHE_FOUND ccache)
    find_program(GGML_SCCACHE_FOUND sccache)

    if (GGML_CCACHE_FOUND OR GGML_SCCACHE_FOUND)
        if(GGML_CCACHE_FOUND)
            set(GGML_CCACHE_VARIANT ccache)
        else()
            set(GGML_CCACHE_VARIANT sccache)
        endif()
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${GGML_CCACHE_VARIANT}")
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "${GGML_CCACHE_VARIANT} found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
endif()

# POSIX conformance
add_compile_definitions(_XOPEN_SOURCE=600)
add_compile_definitions(_GNU_SOURCE)

# ggml

if (GGML_BACKEND_DL AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "GGML_BACKEND_DL requires BUILD_SHARED_LIBS")
endif()

add_library(ggml-base
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ../include/ggml-opt.h
            ../include/gguf.h
            ggml.c
            ggml.cpp
            ggml-alloc.c
            ggml-backend.cpp
            ggml-backend-meta.cpp
            ggml-opt.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-quants.c
            ggml-quants.h
            gguf.cpp)

set_target_properties(ggml-base PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

target_include_directories(ggml-base PRIVATE .)
if (GGML_BACKEND_DL)
    target_compile_definitions(ggml-base PUBLIC GGML_BACKEND_DL)
endif()

if (GGML_SCHED_NO_REALLOC)
    target_compile_definitions(ggml-base PUBLIC GGML_SCHED_NO_REALLOC)
endif()

add_library(ggml
            ggml-backend-dl.cpp
            ggml-backend-reg.cpp)
add_library(ggml::ggml ALIAS ggml)

set_target_properties(ggml PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

if (GGML_BACKEND_DIR)
    if (NOT GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_BACKEND_DIR requires GGML_BACKEND_DL")
    endif()
    target_compile_definitions(ggml PUBLIC GGML_BACKEND_DIR="${GGML_BACKEND_DIR}")
endif()

target_link_libraries(ggml PUBLIC ggml-base)
target_link_libraries(ggml PRIVATE dl)

function(ggml_add_backend_library backend)
    if (GGML_BACKEND_DL)
        add_library(${backend} MODULE ${ARGN})
        set_target_properties(${backend} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_DL)
        add_dependencies(ggml ${backend})
        if (GGML_BACKEND_DIR)
            install(TARGETS ${backend} LIBRARY DESTINATION ${GGML_BACKEND_DIR})
        else()
            install(TARGETS ${backend} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif()
    else()
        add_library(${backend} ${ARGN})
        target_link_libraries(ggml PUBLIC ${backend})
        install(TARGETS ${backend} LIBRARY)
    endif()

    target_link_libraries(${backend} PRIVATE ggml-base)
    target_include_directories(${backend} PRIVATE ..)

    if (${BUILD_SHARED_LIBS})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_BUILD)
        target_compile_definitions(${backend} PUBLIC  GGML_BACKEND_SHARED)
    endif()

    set_target_properties(${backend} PROPERTIES
        VERSION ${GGML_VERSION}
        SOVERSION ${GGML_VERSION_MAJOR}
    )

    if(NOT GGML_AVAILABLE_BACKENDS)
        set(GGML_AVAILABLE_BACKENDS "${backend}"
            CACHE INTERNAL "List of backends for cmake package")
    else()
        list(FIND GGML_AVAILABLE_BACKENDS "${backend}" has_backend)
        if(has_backend EQUAL -1)
            set(GGML_AVAILABLE_BACKENDS "${GGML_AVAILABLE_BACKENDS};${backend}"
                CACHE INTERNAL "List of backends for cmake package")
        endif()
    endif()
endfunction()

function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        string(TOLOWER "ggml-${backend}" backend_target)
        add_subdirectory(${backend_target})
        message(STATUS "Including ${backend} backend")
        if (NOT GGML_BACKEND_DL)
            string(TOUPPER "GGML_USE_${backend}" backend_use)
            target_compile_definitions(ggml PUBLIC ${backend_use})
        endif()
    endif()
endfunction()

function(ggml_add_cpu_backend_variant tag_name)
    set(GGML_CPU_TAG_NAME ${tag_name})
    foreach (feat NATIVE
                  SSE42
                  AVX AVX2 BMI2 AVX_VNNI FMA F16C
                  AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16
                  AMX_TILE AMX_INT8 AMX_BF16)
        set(GGML_${feat} OFF)
    endforeach()

    foreach (feat ${ARGN})
        set(GGML_${feat} ON)
    endforeach()

    ggml_add_cpu_backend_variant_impl(${tag_name})
endfunction()

ggml_add_backend(CPU)

if (GGML_CPU_ALL_VARIANTS)
    if (NOT GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL")
    endif()
    ggml_add_cpu_backend_variant(x64)
    ggml_add_cpu_backend_variant(sse42              SSE42)
    ggml_add_cpu_backend_variant(sandybridge        SSE42 AVX)
    ggml_add_cpu_backend_variant(ivybridge      SSE42 AVX F16C)
    ggml_add_cpu_backend_variant(piledriver     SSE42 AVX F16C FMA)
    ggml_add_cpu_backend_variant(haswell            SSE42 AVX F16C FMA AVX2 BMI2)
    ggml_add_cpu_backend_variant(skylakex           SSE42 AVX F16C FMA AVX2 BMI2 AVX512)
    ggml_add_cpu_backend_variant(cannonlake         SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI)
    ggml_add_cpu_backend_variant(cascadelake        SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI)
    ggml_add_cpu_backend_variant(icelake            SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI)
    ggml_add_cpu_backend_variant(cooperlake     SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI AVX512_BF16)
    ggml_add_cpu_backend_variant(zen4           SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
    ggml_add_cpu_backend_variant(alderlake          SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
    ggml_add_cpu_backend_variant(sapphirerapids SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
elseif (GGML_CPU)
    ggml_add_cpu_backend_variant_impl("")
endif()

ggml_add_backend(CUDA)

foreach (target ggml-base ggml)
    target_include_directories(${target} PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<INSTALL_INTERFACE:include>)
    target_compile_features   (${target} PRIVATE c_std_11 cxx_std_17)
endforeach()

target_link_libraries(ggml-base PRIVATE Threads::Threads)
target_link_libraries(ggml-base PRIVATE m)

if (BUILD_SHARED_LIBS)
    foreach (target ggml-base ggml)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(${target} PRIVATE GGML_BUILD)
        target_compile_definitions(${target} PUBLIC  GGML_SHARED)
    endforeach()
endif()
