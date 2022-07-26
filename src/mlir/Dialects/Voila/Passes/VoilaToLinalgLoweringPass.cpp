#include "mlir/Dialects/Voila/Passes/VoilaToLinalgLoweringPass.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialects/Voila/IR/VoilaDialect.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialects/Voila/lowering/ArithmeticOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/ComparisonOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/ConstOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/GatherOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/HashOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/LogicalOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/LookupOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/NotOpLowering.hpp"
namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::voila;
    namespace lowering
    {
        void VoilaToLinalgLoweringPass::runOnOperation()
        {
            auto function = getOperation();

            // We only lower the main function as we expect that all other functions have
            // been inlined.
            if (function.getName() != "main")
                return;

            // The first thing to define is the conversion target. This will define the
            // final target for this lowering.
            ConversionTarget target(getContext());

            // We define the specific operations, or dialects, that are legal targets for
            // this lowering.
            target.addLegalDialect<BuiltinDialect, AffineDialect, memref::MemRefDialect, func::FuncDialect,
                                   linalg::LinalgDialect, tensor::TensorDialect, scf::SCFDialect, vector::VectorDialect,
                                   arith::ArithmeticDialect, bufferization::BufferizationDialect>();

            // We also define the dialect as Illegal so that the conversion will fail
            // if any of these operations are *not* converted. Given that we actually want
            // a partial lowering, we explicitly mark the Toy operations that don't want
            // to lower, `toy.print`, as `legal`.
            target.addIllegalDialect<VoilaDialect>();
            target.addLegalOp<EmitOp, ::mlir::voila::SelectOp, ReadOp, ScatterOp, LoopOp, SumOp, CountOp, MinOp, MaxOp,
                              AvgOp, InsertOp>();
            // Now that the conversion target has been defined, we just need to provide
            // the set of patterns that will lower the Toy operations.
            RewritePatternSet patterns(&getContext());
            patterns.add<AndOpLowering, OrOpLowering, NotOpLowering, AddOpLowering, SubOpLowering, MulOpLowering,
                         DivOpLowering, ModOpLowering, EqOpLowering, NeqOpLowering, LeOpLowering, LeqOpLowering,
                         GeOpLowering, GeqOpLowering, HashOpLowering, LookupOpLowering, BoolConstOpLowering,
                         IntConstOpLowering, FltConstOpLowering, GatherOpLowering>(&getContext());

            // With the target and rewrite patterns defined, we can now attempt the
            // conversion. The conversion will signal failure if any of our `illegal`
            // operations were not converted successfully.
            if (failed(applyPartialConversion(function, target, std::move(patterns))))
            {
                function.dump();
                signalPassFailure();
            }
        }
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createLowerToLinalgPass()
    {
        return std::make_unique<lowering::VoilaToLinalgLoweringPass>();
    }
} // namespace voila::mlir