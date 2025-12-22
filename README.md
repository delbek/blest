# BLEST: Blazingly Efficient BFS using Tensor Cores

1. In `CMakeLists.txt`, set the target GPU architecture to match your hardware, e.g.
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 90)
   ```
   Adjust 90 (SM 90: Hopper, SM 80: Ampere, ...) to the appropriate value of your GPU. (Learn your GPU architecture from: [architecture](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/))

2. Build:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run
   ```bash
   ./blest -d <directory> -g <graph_name> -j <0|1> -w <window_size>
   ```
   Argument Explanations:

   1. -d: Absolute directory path under which your BFS source files (ending with .txt) are located and to which BLEST will dump 
       results and intermediate files (e.g, /home/blest/intermediate/)

   2. -g: Graph name (e.g, GAP-twitter)

   3. -j: Jaccard enabled (0 or 1) -- WE RECOMMEND THIS TO BE SET TO 1

   4. -w: Window size (an unsigned integer) -- WE RECOMMEND THIS TO BE SET TO 65536

   Note: You do not need to download graphs as the library will download it for you if it finds in SuiteSparse.

   Note: An example BFS source file for GAP-twitter is shared with you under intermediate/ directory.

   Note: If you change the ordering arguments (-j or -w) between runs, delete any cached .bin files in the directory specified by -d. Otherwise, BLEST may reuse the old cache and ignore your new ordering settings.
