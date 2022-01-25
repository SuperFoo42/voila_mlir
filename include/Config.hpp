#pragma once

#include <magic_enum.hpp>
#include <string>
#include <thread>
namespace cxxopts
{
    class ParseResult;
}

namespace voila
{
    class Program;
    class Config
    {
        friend class Program;
        bool _debug : 1;
        bool _optimize : 1;
        bool _fuse : 1;
        bool _tile : 1;
        bool _peel : 1;
        bool _vectorize : 1;
        bool _vectorize_reductions : 1;
        bool _parallelize : 1;
        bool _parallelize_reductions : 1;
        bool _async_parallel : 1;
        bool _gpu_parallel : 1;
        bool _openmp_parallel : 1;
        bool _unroll : 1;
        bool _profile : 1;
        bool _plotAST : 1;
        bool _printMLIR : 1;
        bool _printLoweredMLIR : 1;
        bool _printLLVM : 1;
        int64_t _tile_size;
        int32_t _vector_size;
        int32_t _parallel_threads;
        int32_t _ht_size_factor;

        std::string ASTOutFile;
        std::string MLIROutFile;
        std::string MLIRLoweredOutFile;
        std::string LLVMOutFile;

      public:
        explicit Config(cxxopts::ParseResult &opts);
        explicit Config(bool debug = false,
                        bool optimize = true,
                        bool fuse = true,
                        bool tile = true,
                        bool peel = true,
                        bool vectorize = true,
                        bool vectorizeReductions = true,
                        bool parallelize = true,
                        bool parallelizeReductions = true,
                        bool cpu_parallel = true,
                        bool gpu_parallel = false,
                        bool openmp_parallel = false,
                        bool unroll = true,
                        bool profile = true,
                        bool plotAst = true,
                        bool printMlir = true,
                        bool printLoweredMlir = true,
                        bool printLlvm = true,
                        int64_t tileSize = -1,
                        int32_t vectorSize = 8,
                        int32_t parallelThreads = std::max<int32_t>(std::thread::hardware_concurrency(), 1),
                        int32_t htSizeFactor = -1,
                        std::string astOutFile = "",
                        std::string mlirOutFile = "",
                        std::string mlirLoweredOutFile = "",
                        std::string llvmOutFile = "");
        Config &debug(bool flag = true);
        Config &optimize(bool flag = true);
        Config &fuse(bool flag = true);
        Config &tile(bool flag = true);
        Config &peel(bool flag = true);
        Config &vectorize(bool flag = true);
        Config &vectorize_reductions(bool flag = true);
        Config &parallelize(bool flag = true);
        Config &parallelize_reductions(bool flag = true);
        Config &unroll(bool flag = true);
        Config &profile(bool flag = true);
        Config &plot_ast(bool flag = true);
        Config &print_mlir(bool flag = true);
        Config &print_lowered_mlir(bool flag = true);
        Config &print_llvm(bool flag = true);
        Config &tile_size(int64_t tileSize = -1);
        Config &vector_size(int32_t vectorSize = 8);
        Config &parallel_threads(int32_t parallelThreads = std::thread::hardware_concurrency());
        Config &ht_size_factor(int32_t htSizeFactor = -1);
        Config &ast_out_file(const std::string &astOutFile);
        Config &mlir_out_file(std::string mlirOutFile);
        Config &mlir_lowered_out_file(std::string mlirLoweredOutFile);
        Config &llvm_out_file(std::string llvmOutFile);
        Config &gpu_parallel(bool flag = true);
        Config &async_parallel(bool flag = true);
        Config &openmp_parallel(bool flag = true);
    };
} // namespace voila
