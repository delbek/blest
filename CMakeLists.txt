cmake_minimum_required(VERSION 3.18)
project(sutensor)

# C++
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CUDA_OPENMP_FLAGS "${OpenMP_CXX_FLAGS}")
    if(OpenMP_CXX_FLAGS MATCHES "-mp")
        set(CUDA_OPENMP_FLAGS "-fopenmp")
    endif()
    separate_arguments(CUDA_OPENMP_FLAGS_LIST NATIVE_COMMAND "${CUDA_OPENMP_FLAGS}")
    foreach(flag IN LISTS CUDA_OPENMP_FLAGS_LIST)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${flag}")
    endforeach()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 90)
    enable_language(CUDA)
endif()

set(METIS_ROOT "$ENV{HOME}/local")
if(NOT EXISTS "${METIS_ROOT}")
    message(FATAL_ERROR "Expected METIS installation under ${METIS_ROOT}")
endif()

set(METIS_INCLUDE_DIR "${METIS_ROOT}/include")
set(METIS_LIB_DIRS
    "${METIS_ROOT}/lib"
    "${METIS_ROOT}/lib64"
)

find_library(METIS_LIBRARY metis HINTS ${METIS_LIB_DIRS})
find_library(GKLIB_LIBRARY GKlib HINTS ${METIS_LIB_DIRS})

if(NOT METIS_LIBRARY)
    message(FATAL_ERROR "METIS library not found; searched ${METIS_LIB_DIRS}")
endif()

if(NOT GKLIB_LIBRARY)
    message(FATAL_ERROR "GKlib library not found; searched ${METIS_LIB_DIRS}")
endif()

include_directories(.)
include_directories(Gorder)

add_executable(sutensor
    main.cu
    # Gorder
    Gorder/Graph.cpp
    Gorder/Util.cpp
    Gorder/UnitHeap.cpp
    # Gorder
)

target_link_libraries(sutensor
    OpenMP::OpenMP_CXX
    cuda
    CUDA::cudart
    ${CUDA_LIBRARIES}
    ${METIS_LIBRARY}
    ${GKLIB_LIBRARY}
)

target_include_directories(sutensor PRIVATE
    ${METIS_INCLUDE_DIR}
)

target_compile_options(sutensor PRIVATE
    -O3 -g
    $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --std=c++20
    --generate-line-info
    >
)
