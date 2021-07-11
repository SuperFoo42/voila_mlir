#pragma once

#include <string>

namespace cxxopts {
    class ParseResult;
}

namespace voila
{
    struct Config
    {
        bool debug;
        bool optimize;
        bool plotAST;
        bool printMLIR;
        bool printLoweredMLIR;
        bool printLLVM;
        std::string ASTOutFile;
        std::string MLIROutFile;
        std::string MLIRLoweredOutFile;
        std::string LLVMOutFile;

        Config();

        explicit Config(cxxopts::ParseResult &opts);
    };
} // namespace voila
