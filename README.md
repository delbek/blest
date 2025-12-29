# BLEST: Blazingly Efficient BFS using Tensor Cores

BLEST is a state-of-the-art library to execute Breadth First Search (BFS) on modern GPUs. Its novel data structure, combined with unprecedented compute mechanics, delivers immense performance across massive real-world graphs on massively parallel processors, making it one of the most competitive BFS frameworks ever proposed to date.

If you use this library in your research, please cite:
<br/><br/>
Deniz Elbek and Kamer Kaya. BLEST: Blazingly Efficient BFS using Tensor Cores (2025). arXiv:2512.21967. URL https://www.arxiv.org/abs/2512.21967

---

### Hard Requirements

| **Requirement** | **Minimum Version** |
| --------------- |---------------------|
| C++             | `20`                |
| G++             | `12.3.0`            |
| CMake           | `3.18`              |
| CUDA            | `13.0`              |
| GPU Compute Cap.| `80`                |
| libcurl         |  -                  |
| Linux OS        |  -                  |

---

## Step-by-Step Guide

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

   4. -w: Window size (an unsigned integer) -- WE RECOMMEND THIS TO BE SET AT LEAST TO 65536

   Note: You do not need to download graphs as the library will download it for you if it finds in SuiteSparse.

   Note: An example BFS source file for GAP-twitter is shared with you under intermediate/ directory.

   Note: If you change the ordering arguments (-j or -w) between runs, delete any cached .bin files in the directory specified by -d. Otherwise, BLEST may reuse the old cache and ignore your new ordering settings.
