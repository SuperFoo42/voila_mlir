#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/VoilaDialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#pragma GCC diagnostic pop

#include "BenchmarkState.hpp"
#include "QueryGenerator.hpp"

#include <cxxopts.hpp>
#include <iostream>
#include <random>
#include <spdlog/spdlog.h>

std::unique_ptr<BenchmarkState> benchmarkState = nullptr;
std::unique_ptr<QueryGenerator> queryGenerator = nullptr;
int iterations = 10;

int main(int argc, char **argv)
{
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registry.insert<mlir::voila::VoilaDialect>();
    registry.insert<mlir::StandardOpsDialect>();
    ::mlir::registerAllDialects(registry);
    mlir::registerMLIRContextCLOptions();
    //spdlog::set_level(spdlog::level::debug);

    ::benchmark::Initialize(&argc, argv);

    cxxopts::Options options("TPC-H Benchmark for Voila");
    options.add_options()("h, help", "Show help")(
        "d, data-path", "Path to tpc-h benchmark data set with compressed, uncompressed and denormalized files",
        cxxopts::value<std::string>())(
        "s, seed", "Random int to seed the random number generator",
        cxxopts::value<std::mt19937::result_type>()->default_value(std::to_string(std::mt19937::default_seed)))(
        "i, iterations", "Number of iterations for each bechmark", cxxopts::value<int>()->default_value("10"));

    try
    {
        auto cmd = options.parse(argc, argv);

        benchmarkState = std::make_unique<BenchmarkState>(cmd["d"].as<std::string>());
        queryGenerator = std::make_unique<QueryGenerator>(cmd["s"].as<std::mt19937::result_type>(), cmd["i"].as<int>());
        iterations = cmd["i"].as<int>(); //TODO: this is initialized after the benchmarks, so it will never be used
        ::benchmark::RunSpecifiedBenchmarks();
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        std::abort();
    }
}