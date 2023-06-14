#include "mlir/Dialects/Voila/Passes/VoilaToAffineLoweringPass.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"                         // for Ari...
#include "mlir/Dialect/Func/IR/FuncOps.h"                        // for FuncOp
#include "mlir/Dialect/Tensor/IR/Tensor.h"                      // for Ten...
#include "mlir/Dialects/Voila/IR/VoilaDialect.h"                    // for Voi...
#include "mlir/Dialects/Voila/lowering/ArithmeticOpLowering.hpp" // for Add...
#include "mlir/Dialects/Voila/lowering/AvgOpLowering.hpp"        // for Avg...
#include "mlir/Dialects/Voila/lowering/ComparisonOpLowering.hpp" // for EqO...
#include "mlir/Dialects/Voila/lowering/CountOpLowering.hpp"      // for Cou...
#include "mlir/Dialects/Voila/lowering/EmitOpLowering.hpp"       // for Emi...
#include "mlir/Dialects/Voila/lowering/InsertOpLowering.hpp"     // for Ins...
#include "mlir/Dialects/Voila/lowering/LogicalOpLowering.hpp"    // for And...
#include "mlir/Dialects/Voila/lowering/LoopOpLowering.hpp"       // for Loo...
#include "mlir/Dialects/Voila/lowering/MaxOpLowering.hpp"        // for Max...
#include "mlir/Dialects/Voila/lowering/MinOpLowering.hpp"        // for Min...
#include "mlir/Dialects/Voila/lowering/ReadOpLowering.hpp"       // for Rea...
#include "mlir/Dialects/Voila/lowering/ScatterOpLowering.hpp"    // for Sca...
#include "mlir/Dialects/Voila/lowering/SelectOpLowering.hpp"     // for Sel...
#include "mlir/Dialects/Voila/lowering/SumOpLowering.hpp"        // for Sum...
#include "mlir/IR/Block.h"                                       // for Block
#include "mlir/IR/BuiltinDialect.h"                              // for Bui...
#include "mlir/IR/BuiltinTypes.h"                                // for Fun...
#include "mlir/IR/Operation.h"                                   // for Ope...
#include "mlir/IR/PatternMatch.h"                                // for Rew...
#include "mlir/IR/Region.h"                                      // for Region
#include "mlir/Support/LLVM.h"                                   // for mlir
#include "mlir/Support/LogicalResult.h"                          // for failed
#include "mlir/Transforms/DialectConversion.h"                   // for Con...
#include <algorithm>                                             // for max
#include <utility>                                               // for move

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
            target.addLegalDialect<BuiltinDialect, affine::AffineDialect, memref::MemRefDialect, func::FuncDialect,
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
            patterns.add<SelectOpLowering, ReadOpLowering, ScatterOpLowering, LoopOpLowering, InsertOpLowering,
                         SumOpLowering, CountOpLowering, MinOpLowering, MaxOpLowering, AvgOpLowering>(&getContext());
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

    std::unique_ptr<Pass> createLowerToAffinePass() { return std::make_unique<lowering::VoilaToAffineLoweringPass>(); }
} // namespace voila::mlir
