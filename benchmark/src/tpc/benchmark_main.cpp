#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/IR/VoilaDialect.h>
#pragma GCC diagnostic pop

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
    //TODO: add argument for benchmark data path
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
}