#include "Config.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <cxxopts.hpp>
#include <utility>
#pragma GCC diagnostic pop

namespace voila
{
    Config::Config(cxxopts::ParseResult &opts) :
        Config(opts.count("v"),
               opts.count("O"),
               true,
               true,
               true,
               true,
               true,
               true,
               true,
               true,
               false,
               false,
               true,
               true,
               opts.count("a"),
               opts.count("d"),
               opts.count("l"),
               opts.count("O"),
               -1,
               8,
               std::max<int32_t>(std::thread::hardware_concurrency(), 1),
               -1,
               opts["f"].as<std::string>(),
               opts["f"].as<std::string>(),
               opts["f"].as<std::string>(),
               opts["f"].as<std::string>())
    {
    }

    Config::Config(bool debug,
                   bool optimize,
                   bool fuse,
                   bool tile,
                   bool peel,
                   bool vectorize,
                   bool vectorizeReductions,
                   bool parallelize,
                   bool parallelizeReductions,
                   bool cpu_parallel,
                   bool gpu_parallel,
                   bool openmp_parallel,
                   bool unroll,
                   bool profile,
                   bool plotAst,
                   bool printMlir,
                   bool printLoweredMlir,
                   bool printLlvm,
                   int64_t tileSize,
                   int32_t vectorSize,
                   int32_t parallelThreads,
                   int32_t htSizeFactor,
                   std::string astOutFile,
                   std::string mlirOutFile,
                   std::string mlirLoweredOutFile,
                   std::string llvmOutFile) :
        _debug(debug),
        _optimize(optimize),
        _fuse(fuse),
        _tile(tile),
        _peel(peel),
        _vectorize(vectorize),
        _vectorize_reductions(vectorizeReductions),
        _parallelize(parallelize),
        _parallelize_reductions(parallelizeReductions),
        _async_parallel(cpu_parallel),
        _gpu_parallel(gpu_parallel),
        _openmp_parallel(openmp_parallel),
        _unroll(unroll),
        _profile(profile),
        _plotAST(plotAst),
        _printMLIR(printMlir),
        _printLoweredMLIR(printLoweredMlir),
        _printLLVM(printLlvm),
        _tile_size(tileSize),
        _vector_size(vectorSize),
        _parallel_threads(parallelThreads),
        _ht_size_factor(htSizeFactor),
        ASTOutFile(std::move(astOutFile)),
        MLIROutFile(std::move(mlirOutFile)),
        MLIRLoweredOutFile(std::move(mlirLoweredOutFile)),
        LLVMOutFile(std::move(llvmOutFile))
    {
        // TODO: check conflicting options
    }
    Config &Config::debug(bool flag)
    {
        Config::_debug = flag;
        return *this;
    }
    Config &Config::optimize(bool flag)
    {
        Config::_optimize = flag;
        return *this;
    }
    Config &Config::fuse(bool flag)
    {
        Config::_fuse = flag;
        return *this;
    }
    Config &Config::tile(bool flag)
    {
        Config::_tile = flag;
        return *this;
    }
    Config &Config::peel(bool flag)
    {
        Config::_peel = flag;
        return *this;
    }
    Config &Config::vectorize(bool flag)
    {
        Config::_vectorize = flag;
        return *this;
    }
    Config &Config::vectorize_reductions(bool flag)
    {
        _vectorize_reductions = flag;
        return *this;
    }
    Config &Config::parallelize(bool flag)
    {
        Config::_parallelize = flag;
        return *this;
    }
    Config &Config::parallelize_reductions(bool flag)
    {
        _parallelize_reductions = flag;
        return *this;
    }
    Config &Config::async_parallel(bool flag)
    {
        _async_parallel = flag;
        return *this;
    }
    Config &Config::openmp_parallel(bool flag)
    {
        _openmp_parallel = flag;
        return *this;
    }
    Config &Config::gpu_parallel(bool flag)
    {
        _gpu_parallel = flag;
        return *this;
    }
    Config &Config::unroll(bool flag)
    {
        Config::_unroll = flag;
        return *this;
    }
    Config &Config::profile(bool flag)
    {
        Config::_profile = flag;
        return *this;
    }
    Config &Config::plot_ast(bool flag)
    {
        _plotAST = flag;
        return *this;
    }
    Config &Config::print_mlir(bool flag)
    {
        _printMLIR = flag;
        return *this;
    }
    Config &Config::print_lowered_mlir(bool flag)
    {
        _printLoweredMLIR = flag;
        return *this;
    }
    Config &Config::print_llvm(bool flag)
    {
        _printLLVM = flag;
        return *this;
    }
    Config &Config::tile_size(int64_t tileSize)
    {
        _tile_size = tileSize;
        return *this;
    }
    Config &Config::vector_size(int32_t vectorSize)
    {
        _vector_size = vectorSize;
        return *this;
    }
    Config &Config::parallel_threads(int32_t parallelThreads)
    {
        _parallel_threads = parallelThreads;
        return *this;
    }
    Config &Config::ht_size_factor(int32_t htSizeFactor)
    {
        _ht_size_factor = htSizeFactor;
        return *this;
    }
    Config &Config::ast_out_file(const std::string &astOutFile)
    {
        ASTOutFile = astOutFile;
        return *this;
    }
    Config &Config::mlir_out_file(std::string mlirOutFile)
    {
        MLIROutFile = std::move(mlirOutFile);
        return *this;
    }
    Config &Config::mlir_lowered_out_file(std::string mlirLoweredOutFile)
    {
        MLIRLoweredOutFile = std::move(mlirLoweredOutFile);
        return *this;
    }
    Config &Config::llvm_out_file(std::string llvmOutFile)
    {
        LLVMOutFile = std::move(llvmOutFile);
        return *this;
    }
} // namespace voila