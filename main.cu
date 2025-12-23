#include "Benchmark.cuh"

static void print_usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog
        << " -d <directory> -g <graph_name> -j <0|1> -w <window_size>\n"
        << "  -d \t Absolute directory path under which your BFS source files are located and to which BLEST will dump results and intermediate files (e.g, /home/blest/intermediate/)\n"
        << "  -g \t Graph name (e.g, GAP-twitter)\n"
        << "  -j \t Jaccard enabled (0 or 1) -- WE RECOMMEND THIS TO BE SET TO 1\n"
        << "  -w \t Window size (an unsigned integer) -- WE RECOMMEND THIS TO BE SET TO 65536\n";
}

static Config parse_args(int argc, char** argv)
{
    std::string directory;
    std::string matrixName;
    bool jackardEnabled_set = false;
    bool jackardEnabled = false;
    bool windowSize_set = false;
    unsigned windowSize = 0;

    auto require_value = [&](int& i, const char* opt) -> std::string
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
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "-d")
        {
            directory = require_value(i, "-d");
        }
        else if (arg == "-g")
        {
            matrixName = require_value(i, "-g");
        }
        else if (arg == "-j")
        {
            const std::string v = require_value(i, "-j");
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
            const std::string v = require_value(i, "-w");
            try
            {
                unsigned long tmp = std::stoul(v);
                if (tmp > static_cast<unsigned long>(std::numeric_limits<unsigned>::max()))
                {
                    throw std::out_of_range("window size too large");
                }
                windowSize = static_cast<unsigned>(tmp);
            }
            catch (const std::exception&)
            {
                throw std::runtime_error("Invalid value for -w (expected unsigned integer), got: " + v);
            }
            windowSize_set = true;
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (directory.empty()) throw std::runtime_error("Missing required option: -d <directory>");
    if (matrixName.empty()) throw std::runtime_error("Missing required option: -g <graph_name>");
    if (!jackardEnabled_set) throw std::runtime_error("Missing required option: -j <0|1>");
    if (!windowSize_set) throw std::runtime_error("Missing required option: -w <window_size>");

    return Config{directory, matrixName, jackardEnabled, windowSize};
}

int main(int argc, char** argv)
{
    try
    {
        Config config = parse_args(argc, argv);
        Benchmark benchmark;
        benchmark.main(config);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}
