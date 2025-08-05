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
    add_definitions(-DGPU_AVAILABLE)
endif()

# MPI
find_package(MPI QUIET)
if(MPI_FOUND)
    add_definitions(-DMPI_AVAILABLE)
    include_directories(${MPI_INCLUDE_PATH})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
endif()

include_directories(.)

add_executable(sutensor
        main.cu
)

target_compile_options(sutensor PRIVATE
    -O3
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90;-O3;--generate-line-info>
)

target_link_libraries(sutensor
        OpenMP::OpenMP_CXX
        cuda
        CUDA::cudart
        ${CUDA_LIBRARIES}
)

if(MPI_FOUND)
    target_link_libraries(sutensor
            MPI::MPI_CXX
    )
endif()
