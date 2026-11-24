cmake_minimum_required(VERSION 3.20)
project(tinyinfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TINYINFER_BUILD_TESTS "Build C++ unit tests" ON)
option(TINYINFER_BUILD_BENCH "Build C++ microbenchmark binary" ON)
option(TINYINFER_BUILD_CLI "Build standalone C++ inference CLI" ON)
option(TINYINFER_BUILD_PYTHON "Build nanobind Python extension" OFF)
option(TINYINFER_ENABLE_OPENMP "Enable OpenMP multithreading" ON)
option(TINYINFER_ENABLE_AVX2 "Enable AVX2+FMA SIMD instructions on x86_64" ON)

add_library(tinyinfer_core STATIC
    src/quant.cc
    src/matmul.cc
    src/ops.cc
    src/kv_cache.cc
    src/attention.cc
    src/mmap_loader.cc
    src/transformer.cc
)
set_target_properties(tinyinfer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tinyinfer_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tinyinfer_core PRIVATE -Wall -Wextra -Wpedantic)
    if(TINYINFER_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(tinyinfer_core PRIVATE -mavx2 -mfma)
    endif()
endif()

if(TINYINFER_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(tinyinfer_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

if(TINYINFER_BUILD_TESTS)
    enable_testing()
    add_executable(tinyinfer_tests tests/cpp/test_kernels.cc)
    target_link_libraries(tinyinfer_tests PRIVATE tinyinfer_core)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(tinyinfer_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME cpp_kernel_tests COMMAND tinyinfer_tests)

    add_executable(tinyinfer_attention_tests tests/cpp/test_attention.cc)
    target_link_libraries(tinyinfer_attention_tests PRIVATE tinyinfer_core)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(tinyinfer_attention_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME cpp_attention_tests COMMAND tinyinfer_attention_tests)

    add_executable(tinyinfer_transformer_tests tests/cpp/test_transformer.cc)
    target_link_libraries(tinyinfer_transformer_tests PRIVATE tinyinfer_core)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(tinyinfer_transformer_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME cpp_transformer_tests COMMAND tinyinfer_transformer_tests)
endif()

if(TINYINFER_BUILD_BENCH)
    add_executable(tinyinfer_bench benchmarks/bench_kernels.cc)
    target_link_libraries(tinyinfer_bench PRIVATE tinyinfer_core)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(tinyinfer_bench PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(TINYINFER_BUILD_CLI)
    add_executable(tinyinfer_cli cli/main.cc)
    target_link_libraries(tinyinfer_cli PRIVATE tinyinfer_core)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(tinyinfer_cli PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(TINYINFER_BUILD_PYTHON)
    find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE NB_DIR
    )
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(_core python/bindings.cc)
    target_link_libraries(_core PRIVATE tinyinfer_core)
    install(TARGETS _core LIBRARY DESTINATION tinyinfer)
endif()
