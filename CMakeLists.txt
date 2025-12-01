cmake_minimum_required(VERSION 3.18)
project(sutensor)

# C++
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 90)
    enable_language(CUDA)
endif()

find_package(CURL REQUIRED)

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
    CURL::libcurl
    OpenMP::OpenMP_CXX
    cuda
    CUDA::cudart
    ${CUDA_LIBRARIES}
)

target_compile_options(sutensor PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -g -std=c++20>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info --std=c++20 -Xptxas=-v>
)
