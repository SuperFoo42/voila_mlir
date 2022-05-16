#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include "mlir/IR/VoilaDialect.h"
#include <spdlog/spdlog.h>

#pragma GCC diagnostic pop

int main(int argc, char **argv) {
    // setup mlir
    //llvm::DebugFlag = true;
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registry.insert<mlir::voila::VoilaDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<mlir::vector::VectorDialect>();
    registry.insert<mlir::omp::OpenMPDialect>();
    ::mlir::registerAllDialects(registry);
    mlir::registerMLIRContextCLOptions();
spdlog::set_level(spdlog::level::debug);
    // setup gtest
    ::testing::InitGoogleTest(&argc, argv);

    // run gtests
    return RUN_ALL_TESTS();
}