#include "Config.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <cxxopts.hpp>
#pragma GCC diagnostic pop
#include <thread>
namespace voila
{
    Config::Config() :
        debug(false),
        optimize(true),
        fuse(true),
        tile(true),
        peel(true),
        vectorize(true),
        vectorize_reductions(true),
        parallelize(true),
        parallelize_reductions(true),
        unroll(true),
        profile(true),
        plotAST(false),
        printMLIR(false),
        printLoweredMLIR(false),
        printLLVM(false),
        tile_size(-1),//-1 is interference from data size
        vector_size(8),//TODO: what is a good size for this?
        parallel_threads(std::max<int32_t>(std::thread::hardware_concurrency(), 1)),
        ASTOutFile(),
        MLIROutFile(),
        MLIRLoweredOutFile(),
        LLVMOutFile()
    {
    }

    Config::Config(cxxopts::ParseResult &opts) :
        debug(opts.count("v")),
        optimize(opts.count("O")),
        fuse(true),
        tile(true),
        peel(true),
        vectorize(true),
        vectorize_reductions(true),
        parallelize(true),
        parallelize_reductions(true),
        unroll(true),
        profile(true),
        plotAST(opts.count("a")),
        printMLIR(opts.count("d")),
        printLoweredMLIR(opts.count("l")),
        printLLVM(opts.count("O")),
        tile_size(-1),//-1 is interference from data size
        vector_size(8),//TODO: what is a good size for this?
        parallel_threads(std::max<int32_t>(std::thread::hardware_concurrency(), 1)),
        ASTOutFile(opts["f"].as<std::string>()),
        MLIROutFile(opts["f"].as<std::string>()),
        MLIRLoweredOutFile(opts["f"].as<std::string>()),
        LLVMOutFile(opts["f"].as<std::string>())
    {
    }
} // namespace voila