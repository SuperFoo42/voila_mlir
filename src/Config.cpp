#include "Config.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <cxxopts.hpp>
#pragma GCC diagnostic pop

namespace voila
{
    Config::Config() :
        debug(false),
        optimize(true),
        plotAST(false),
        printMLIR(false),
        printLoweredMLIR(false),
        printLLVM(false),
        ASTOutFile(),
        MLIROutFile(),
        MLIRLoweredOutFile(),
        LLVMOutFile()
    {
    }

    Config::Config(cxxopts::ParseResult &opts) :
        debug(opts.count("v")),
        optimize(opts.count("O")),
        plotAST(opts.count("a")),
        printMLIR(opts.count("d")),
        printLoweredMLIR(opts.count("l")),
        printLLVM(opts.count("O")),
        ASTOutFile(opts["f"].as<std::string>()),
        MLIROutFile(opts["f"].as<std::string>()),
        MLIRLoweredOutFile(opts["f"].as<std::string>()),
        LLVMOutFile(opts["f"].as<std::string>())
    {
    }
} // namespace voila