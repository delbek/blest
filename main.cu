/*
 * This file is part of the BLEST repository: https://github.com/delbek/blest
 * Author: Deniz Elbek
 *
 * Please see the paper:
 * 
 * @article{Elbek2025BLEST,
 *   title   = {BLEST: Blazingly Efficient BFS using Tensor Cores},
 *   author  = {Elbek, Deniz and Kaya, Kamer},
 *   journal = {arXiv preprint arXiv:2512.21967},
 *   year    = {2025},
 *   doi     = {10.48550/arXiv.2512.21967},
 *   url     = {https://www.arxiv.org/abs/2512.21967}
 * }
 */

#include "Benchmark.cuh"

static void printUsage(const char* prog)
{
    std::cerr
        << "Usage: " << prog
        << " -d <directory> -g <graph_name> -j <0|1> -w <window_size> -k <BFS|MBFS|Closeness|CC|WCC>\n"
        << "  -d \t Absolute directory path under which your BFS source files are located and to which BLEST will dump intermediate files (e.g, /home/blest/intermediate/)\n"
        << "  -g \t Graph name (e.g, GAP-twitter)\n"
        << "  -j \t Jaccard enabled (0 or 1) -- We recommend this to be set to 1\n"
        << "  -w \t Window size (an unsigned integer > 0) -- We recommend this to be set at least to 65536\n"
        << "  -k \t Kernel name: BFS, MBFS, Closeness, CC, or WCC\n";
}

static Config parseArgs(int argc, char** argv)
{
    std::string directory;
    std::string matrixName;
    bool jackardEnabled_set = false;
    bool jackardEnabled = false;
    bool windowSize_set = false;
    unsigned windowSize = 0;
    bool kernelName_set = false;
    std::string kernelName;

    auto requireValue = [&](int& i, const char* opt) -> std::string
    {
        if (i + 1 >= argc)
        {
            throw std::runtime_error(std::string("Missing value for ") + opt);
        }
        ++i;
        return std::string(argv[i]);
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            std::exit(0);
        }
        else if (arg == "-d")
        {
            directory = requireValue(i, "-d");
        }
        else if (arg == "-g")
        {
            matrixName = requireValue(i, "-g");
        }
        else if (arg == "-j")
        {
            const std::string v = requireValue(i, "-j");
            if (v == "0")
            {
                jackardEnabled = false;
            }
            else if (v == "1")
            {
                jackardEnabled = true;
            }
            else
            {
                throw std::runtime_error("Invalid value for -j (expected 0 or 1), got: " + v);
            }
            jackardEnabled_set = true;
        }
        else if (arg == "-w")
        {
            const std::string v = requireValue(i, "-w");
            try
            {
                unsigned long tmp = std::stoul(v);
                windowSize = static_cast<unsigned>(tmp);
                if (windowSize == 0)
                {
                    throw std::exception();
                }
            }
            catch (const std::exception&)
            {
                throw std::runtime_error("Invalid value for -w (expected unsigned integer > 0), got: " + v);
            }
            windowSize_set = true;
        }
        else if (arg == "-k")
        {
            kernelName = requireValue(i, "-k");
            if (kernelName != "BFS" && kernelName != "MBFS" && kernelName != "Closeness" && kernelName != "CC" && kernelName != "WCC")
            {
                throw std::runtime_error("Invalid value for -k (expected BFS, MBFS, Closeness, CC, or WCC), got: " + kernelName);
            }
            kernelName_set = true;
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (directory.empty()) throw std::runtime_error("Missing required option: -d <directory>");
    if (matrixName.empty()) throw std::runtime_error("Missing required option: -g <graph_name>");
    if (!kernelName_set) throw std::runtime_error("Missing required option: -k <BFS|MBFS|Closeness|CC|WCC>");
    if (!jackardEnabled_set) throw std::runtime_error("Missing required option: -j <0|1>");
    if (!windowSize_set) throw std::runtime_error("Missing required option: -w <window_size>");

    return Config{directory, matrixName, jackardEnabled, windowSize, kernelName};
}

int main(int argc, char** argv)
{
    try
    {
        Config config = parseArgs(argc, argv);
        Benchmark benchmark;
        benchmark.main(config);

        return 0;
    }
    catch (const std::exception& e)
    {
        printUsage(argv[0]);
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
