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
                   1,
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
                   bool optimize_selections,
                   bool vectorize,
                   bool vectorizeReductions,
                   bool parallelize,
                   bool parallelizeReductions,
                   bool cpu_parallel,
                   bool gpu_parallel,
                   bool openmp_parallel,
                   bool profile,
                   bool plotAst,
                   bool printMlir,
                   bool printLoweredMlir,
                   bool printLlvm,
                   int64_t tileSize,
                   int32_t vectorSize,
                   int32_t parallelThreads,
                   int32_t unrollFactor,
                   int32_t htSizeFactor,
                   std::string astOutFile,
                   std::string mlirOutFile,
                   std::string mlirLoweredOutFile,
                   std::string llvmOutFile) :
        _debug(debug),
        _optimize(optimize),
        _fuse(fuse),
        _tile(tile),
        _optimize_selections(optimize_selections),
        _vectorize(vectorize),
        _vectorize_reductions(vectorizeReductions),
        _parallelize(parallelize),
        _parallelize_reductions(parallelizeReductions),
        _async_parallel(cpu_parallel),
        _gpu_parallel(gpu_parallel),
        _openmp_parallel(openmp_parallel),
        _profile(profile),
        _plotAST(plotAst),
        _printMLIR(printMlir),
        _printLoweredMLIR(printLoweredMlir),
        _printLLVM(printLlvm),
        _tile_size(tileSize),
        _vector_size(vectorSize),
        _parallel_threads(parallelThreads),
        _unroll_factor(unrollFactor),
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
    Config &Config::optimize_selections(bool flag)
    {
        Config::_optimize_selections = flag;
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
    Config &Config::unroll_factor(int32_t unrollFactor)
    {
        _unroll_factor = unrollFactor;
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

    std::ostream &operator<<(std::ostream &o, const Config &c)
    {
        o << std::boolalpha;
        o << "debug: " << c._debug << "\n";
        o << "optimize: " << c._optimize << "\n";
        o << "fuse: " << c._fuse << "\n";
        o << "tile: " << c._tile << "\n";
        o << "optimize_selections: " << c._optimize_selections << "\n";
        o << "vectorize: " << c._vectorize << "\n";
        o << "vectorize_reductions: " << c._vectorize_reductions << "\n";
        o << "parallelize: " << c._parallelize << "\n";
        o << "parallelize_reductions: " << c._parallelize_reductions << "\n";
        o << "async_parallel: " << c._async_parallel << "\n";
        o << "openmp_parallel: " << c._openmp_parallel << "\n";
        o << "profile: " << c._profile << "\n";
        o << "plot_ast: " << c._plotAST << "\n";
        o << "print_mlir: " << c._printMLIR << "\n";
        o << "print_lowered_mlir: " << c._printLoweredMLIR << "\n";
        o << "print_llvm: " << c._printLLVM << "\n";
        o << "tile_size: " << c._tile_size << "\n";
        o << "vector_size: " << c._vector_size << "\n";
        o << "parallel_threads: " << c._parallel_threads << "\n";
        o << "unroll_factor: " << c._unroll_factor << "\n";
        return o;
    }
} // namespace voila