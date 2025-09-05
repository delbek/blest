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

include_directories(.)

add_executable(sutensor
    main.cu
)

target_link_libraries(sutensor
		OpenMP::OpenMP_CXX
    cuda
    CUDA::cudart
    ${CUDA_LIBRARIES}
)

target_compile_options(sutensor PRIVATE
  -O3 -funroll-loops -finline-functions -march=native -ffast-math
  $<$<COMPILE_LANGUAGE:CUDA>:
		-O3
    --generate-line-info
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --fmad=true
    -Xptxas=-v
    --use_fast_math
  >
)
