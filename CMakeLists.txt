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

# MPI
#find_package(MPI QUIET)
if(MPI_FOUND)
  add_definitions(-DMPI_AVAILABLE)
  include_directories(${MPI_INCLUDE_PATH})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
endif()
#

# CURL
find_package(CURL REQUIRED)
#

include_directories(.)
include_directories(DataStructures)
include_directories(BFS)
include_directories(MBFS)
include_directories(Closeness)
include_directories(Gorder)
include_directories(CC)

add_executable(blest
    main.cu
    # Gorder
    Gorder/Graph.cpp
    Gorder/Util.cpp
    Gorder/UnitHeap.cpp
    #
)

target_link_libraries(blest PRIVATE
    CURL::libcurl
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

if(MPI_FOUND)
  target_link_libraries(blest PRIVATE
    MPI::MPI_CXX
  )
endif()

target_compile_options(blest PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -g -std=c++20>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info --std=c++20 -Xptxas=-v>
)
