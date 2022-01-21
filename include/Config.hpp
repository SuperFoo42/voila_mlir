#pragma once

#include <string>
#include <bitset>
#include <magic_enum.hpp>
namespace cxxopts
{
    class ParseResult;
}

namespace voila
{
    struct Config
    {
        bool debug : 1;
        bool optimize : 1;
        bool fuse : 1;
        bool tile : 1;
        bool peel : 1;
        bool vectorize : 1;
        bool vectorize_reductions : 1;
        bool parallelize : 1;
        bool parallelize_reductions : 1;
        bool unroll : 1;
        bool profile : 1;
        bool plotAST : 1;
        bool printMLIR : 1;
        bool printLoweredMLIR : 1;
        bool printLLVM : 1;
        int64_t tile_size;
        int32_t vector_size;
        int32_t parallel_threads;



        std::string ASTOutFile;
        std::string MLIROutFile;
        std::string MLIRLoweredOutFile;
        std::string LLVMOutFile;

        Config();

        explicit Config(cxxopts::ParseResult &opts);
    };
} // namespace voila
