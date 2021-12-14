#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/VoilaDialect.h"
#include "mlir/lowering/ArithmeticOpLowering.hpp"
#include "mlir/lowering/AvgOpLowering.hpp"
#include "mlir/lowering/ComparisonOpLowering.hpp"
#include "mlir/lowering/ConstOpLowering.hpp"
#include "mlir/lowering/CountOpLowering.hpp"
#include "mlir/lowering/EmitOpLowering.hpp"
#include "mlir/lowering/GatherOpLowering.hpp"
#include "mlir/lowering/HashOpLowering.hpp"
#include "mlir/lowering/LogicalOpLowering.hpp"
#include "mlir/lowering/LoopOpLowering.hpp"
#include "mlir/lowering/MaxOpLowering.hpp"
#include "mlir/lowering/MinOpLowering.hpp"
#include "mlir/lowering/NotOpLowering.hpp"
#include "mlir/lowering/ReadOpLowering.hpp"
#include "mlir/lowering/SelectOpLowering.hpp"
#include "mlir/lowering/SumOpLowering.hpp"
#include "mlir/lowering/LookupOpLowering.hpp"
#include "mlir/lowering/InsertOpLowering.hpp"
#include <memory>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace voila::mlir
{
    namespace lowering
    {
        struct VoilaToLinalgLoweringPass : public ::mlir::PassWrapper<VoilaToLinalgLoweringPass, ::mlir::FunctionPass>
        {
            void getDependentDialects(::mlir::DialectRegistry &registry) const override
            {
                registry
                    .insert<::mlir::StandardOpsDialect, ::mlir::linalg::LinalgDialect, ::mlir::tensor::TensorDialect>();
            }

            void runOnFunction() final;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createLowerToLinalgPass();
} // namespace voila::mlir
