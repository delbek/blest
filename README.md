1. Open `Benchmark.cuh` and set the `MATRIX_DIRECTORY` macro (defined at the top of the file) to your matrix/graph folder, e.g.
   ```cpp
   #define MATRIX_DIRECTORY "YOUR_MATRIX_DIRECTORY/"
   ```

2. Make sure all matrix source files (files that contain the 0-indexed source vertex IDs from which BFS will be initiated) are located under this directory.

3. Update the names of the benchmark graphs in `Benchmark.cuh` to match the matrices you want to run.

4. In `CMakeLists.txt`, set the target GPU architecture to match your hardware, e.g.
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 90)
   ```
   Adjust 90 (SM 90: Hopper, SM 80: Ampere, ...) to the appropriate value of your GPU. (Learn your GPU architecture from: [architecture](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/))

5. Build and run:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ./blest
   ```