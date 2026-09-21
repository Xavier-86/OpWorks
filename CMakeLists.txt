cmake_minimum_required(VERSION 3.24)

# A reproducible default; override for your deployment GPU(s).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
# Prefer nvcc on PATH over a different toolkit next to the host compiler.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(OPWORKS_NVCC nvcc REQUIRED)
    set(CMAKE_CUDA_COMPILER "${OPWORKS_NVCC}" CACHE FILEPATH "CUDA compiler")
endif()
project(OpWorks LANGUAGES CXX CUDA)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.2)
    message(FATAL_ERROR "OpWorks requires CUDA 11.2 or newer")
endif()

add_library(opworks INTERFACE)
add_library(OpWorks::opworks ALIAS opworks)
target_include_directories(opworks INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(opworks INTERFACE cxx_std_17 cuda_std_17)

include(CTest)
if(BUILD_TESTING)
    add_executable(opworks_tests tests/unit/framework.cu tests/unit/multi_tu.cu)
    target_link_libraries(opworks_tests PRIVATE OpWorks::opworks)
    add_test(NAME framework COMMAND opworks_tests)
    set_tests_properties(framework PROPERTIES SKIP_RETURN_CODE 77)

    foreach(op vector_add relu reduction softmax layer_normalization matrix_multiplication)
        add_library(test_${op} SHARED tests/integration/${op}.cu)
        target_link_libraries(test_${op} PRIVATE OpWorks::opworks)
        set_target_properties(test_${op} PROPERTIES
            PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/integration")
    endforeach()
endif()

# Llama inference runtime (opt-in): C ABI shared library plus model tests.
option(OPWORKS_BUILD_LLAMA "Build the Llama inference runtime and model tests" OFF)
if(OPWORKS_BUILD_LLAMA)
    add_library(opworks_llama SHARED src/llama_c_api.cu)
    target_link_libraries(opworks_llama PRIVATE OpWorks::opworks)

    if(BUILD_TESTING)
        add_executable(llama_ops_test tests/model/ops_test.cu)
        target_link_libraries(llama_ops_test PRIVATE OpWorks::opworks)
        add_test(NAME llama_ops
                 COMMAND llama_ops_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama/ops_fixtures.pack")
        set_tests_properties(llama_ops PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(llama_model_test tests/model/llama_test.cu)
        target_link_libraries(llama_model_test PRIVATE OpWorks::opworks)
        add_test(NAME llama_model
                 COMMAND llama_model_test "${CMAKE_CURRENT_SOURCE_DIR}/data/llama-fp32.pack"
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama/model_fixtures.pack"
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama/longctx_fixtures.pack"
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama/regression_fixtures.pack")
        set_tests_properties(llama_model PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(llama_bf16_test tests/model/llama_bf16_test.cu)
        target_link_libraries(llama_bf16_test PRIVATE OpWorks::opworks)
        add_test(NAME llama_bf16
                 COMMAND llama_bf16_test "${CMAKE_CURRENT_SOURCE_DIR}/data/llama-bf16.pack"
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama/model_fixtures.pack"
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama/longctx_fixtures.pack"
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama/regression_fixtures.pack")
        set_tests_properties(llama_bf16 PROPERTIES SKIP_RETURN_CODE 77)

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3")
            set(OPWORKS_MODEL_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3")
        else()
            find_package(Python3 COMPONENTS Interpreter)
            set(OPWORKS_MODEL_PYTHON "${Python3_EXECUTABLE}")
        endif()
        if(OPWORKS_MODEL_PYTHON)
            add_test(NAME llama_tokenizer
                     COMMAND ${OPWORKS_MODEL_PYTHON} "${CMAKE_CURRENT_SOURCE_DIR}/tests/model/test_tokenizer.py")
            set_tests_properties(llama_tokenizer PROPERTIES SKIP_RETURN_CODE 77
                                 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
            add_test(NAME llama_edges
                     COMMAND ${OPWORKS_MODEL_PYTHON} "${CMAKE_CURRENT_SOURCE_DIR}/tests/model/test_edges.py")
            set_tests_properties(llama_edges PROPERTIES SKIP_RETURN_CODE 77
                                 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        endif()
    endif()
endif()
