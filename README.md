# BLEST: Blazingly Efficient BFS using Tensor Cores

BLEST is a state-of-the-art library to execute Breadth First Search (BFS) on modern GPUs. Its novel data structure, combined with unprecedented compute mechanics, delivers immense performance across massive real-world graphs on massively parallel processors, making it one of the most competitive BFS frameworks ever proposed to date.

Please see the paper:

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
   ./blest -d <directory> -g <graph_name> -j <0|1> -w <window_size> -k <BFS|MBFS|Closeness|CC|WCC>
   ```
   Argument Explanations:

   - `-d`: Absolute directory path under which your source files (ending with .txt) are located and to which BLEST will dump intermediate files (e.g, /home/blest/intermediate/)

   - `-g`: Graph name (e.g, GAP-twitter)

   - `-j`: Jaccard enabled (0 or 1) -- We recommend this to be set to 1

   - `-w`: Window size (an unsigned integer > 0) -- We recommend this to be set at least to 65536

   - `-k`: Kernel to execute. Supported values:

     | Kernel      | Description |
     |-------------|-------------|
     | `BFS`       | Executes single-source BFS sequentially for each node listed in `<graph_name>.txt` under `-d`. The file may contain any number of sources. |
     | `MBFS`      | Executes multi-source BFS **concurrently** for all sources listed in `<graph_name>.txt` under `-d`. At most 256 sources are supported. |
     | `Closeness` | Computes the far values (distances) required for closeness centrality for all nodes. Results are written to `<graph_name>_closeness_results.txt` under `-d`. |
     | `CC`        | Finds connected components of the graph. The graph must be undirected. |
     | `WCC`       | Finds weakly connected components of the graph. |

3. Notes

   - You do not need to download graphs as the library will download it for you if it finds in SuiteSparse.

   - An example BFS source file for GAP-twitter is shared with you under intermediate/ directory.

   - Both `BFS` and `MBFS` read source node IDs from a file named `<graph_name>.txt` placed in the directory specified by `-d`. Each line should contain one source node ID. **All source node IDs must be 0-indexed.** `MBFS` supports at most 256 sources; `BFS` has no such restriction.

   - For `CC`, the input graph must be undirected.

   - If you change the ordering arguments (-j or -w) between runs, delete any cached .bin files in the directory specified by -d. Otherwise, BLEST may reuse the old cache and ignore your new ordering settings.