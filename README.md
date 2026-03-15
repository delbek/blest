# BLEST: Blazingly Efficient BFS using Tensor Cores

BLEST is a state-of-the-art library to execute Breadth First Search (BFS) on modern GPUs. Its novel data structure, combined with unprecedented compute mechanics, delivers immense performance across massive real-world graphs on massively parallel processors, making it one of the most competitive BFS frameworks ever proposed to date.

Paper:

```bibtex
@article{Elbek2025BLEST,
  title   = {BLEST: Blazingly Efficient BFS using Tensor Cores},
  author  = {Elbek, Deniz and Kaya, Kamer},
  journal = {arXiv preprint arXiv:2512.21967},
  year    = {2025},
  doi     = {10.48550/arXiv.2512.21967},
  url     = {https://www.arxiv.org/abs/2512.21967}
}
```

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

1. Build:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

2. Run
   ```bash
   ./blest -d <directory> -g <graph_name> -j <0|1> -w <window_size> -k BFS
   ```
   Argument Explanations:

   - -d: Absolute directory path under which your BFS source files (ending with .txt) are located and to which BLEST will dump intermediate files (e.g, /home/blest/intermediate/)

   - -g: Graph name (e.g, GAP-twitter)

   - -j: Jaccard enabled (0 or 1) -- We recommend this to be set to 1

   - -w: Window size (an unsigned integer) -- We recommend this to be set at least to 65536

3. Notes

   - You do not need to download graphs as the library will download it for you if it finds in SuiteSparse.

   - An example BFS source file for GAP-twitter is shared with you under intermediate/ directory.

   - If you change the ordering arguments (-j or -w) between runs, delete any cached .bin files in the directory specified by -d. Otherwise, BLEST may reuse the old cache and ignore your new ordering settings.
