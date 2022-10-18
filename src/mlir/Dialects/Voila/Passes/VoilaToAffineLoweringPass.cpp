#include "mlir/Dialects/Voila/Passes/VoilaToAffineLoweringPass.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialects/Voila/IR/VoilaDialect.h"
#include "mlir/Dialects/Voila/lowering/ArithmeticOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/AvgOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/ComparisonOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/ConstOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/CountOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/EmitOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/InsertOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/LogicalOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/LoopOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/MaxOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/MinOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/ReadOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/SelectOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/SumOpLowering.hpp"
#include "mlir/Dialects/Voila/lowering/ScatterOpLowering.hpp"


namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::voila;
    namespace lowering
    {
        void VoilaToAffineLoweringPass::runOnOperation()
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
            target.addLegalDialect<BuiltinDialect, AffineDialect, memref::MemRefDialect,func::FuncDialect,
                                   linalg::LinalgDialect, scf::SCFDialect, arith::ArithDialect,
                                   bufferization::BufferizationDialect, tensor::TensorDialect>();

            // We also define the dialect as Illegal so that the conversion will fail
            // if any of these operations are *not* converted. Given that we actually want
            // a partial lowering, we explicitly mark the Toy operations that don't want
            // to lower, `toy.print`, as `legal`.
            target.addIllegalDialect<VoilaDialect>();
            // Now that the conversion target has been defined, we just need to provide
            // the set of patterns that will lower the Toy operations.
            RewritePatternSet patterns(&getContext());
            patterns.add<SelectOpLowering, ReadOpLowering,
                         ScatterOpLowering, LoopOpLowering, InsertOpLowering, SumOpLowering, CountOpLowering,
                         MinOpLowering, MaxOpLowering, AvgOpLowering>(&getContext());
            patterns.add<AndOpLowering, OrOpLowering, AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
                         ModOpLowering, EqOpLowering, NeqOpLowering, LeOpLowering, LeqOpLowering, GeOpLowering,
                         GeqOpLowering>(&getContext());

            patterns.add<EmitOpLowering>(&getContext(), function);

            // With the target and rewrite patterns defined, we can now attempt the
            // conversion. The conversion will signal failure if any of our `illegal`
            // operations were not converted successfully.
            if (failed(applyFullConversion(function, target, std::move(patterns))))
            {
                function.dump();
                signalPassFailure();
            }
            // match function return type after emit lowering
            // FIXME: this is only a workaround, there must be a more clean way to achieve bufferization for function
            // return
            auto newType = FunctionType::get(&getContext(), function.getFunctionType().getInputs(),
                                             function.getBody().back().back().getOperandTypes());
            function.setType(newType);
        }
    } // namespace lowering

    std::unique_ptr<Pass> createLowerToAffinePass()
    {
        return std::make_unique<lowering::VoilaToAffineLoweringPass>();
    }
} // namespace voila::mlir
