cmake_minimum_required(VERSION 3.18)
project(blest)

# C++
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
#

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
enable_language(CUDA)
#

# CURL
find_package(CURL REQUIRED)
#

include_directories(.)
include_directories(DataStructures)
include_directories(BFS)
include_directories(Gorder)

add_executable(blest
    main.cu
    # Gorder
    Gorder/Graph.cpp
    Gorder/Util.cpp
    Gorder/UnitHeap.cpp
    #
)

target_link_libraries(blest
    CURL::libcurl
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_options(blest PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -g -std=c++20>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info --std=c++20 -Xptxas=-v>
)
