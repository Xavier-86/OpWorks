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
