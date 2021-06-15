#include "mlir/lowering/VoilaToAffineLoweringPass.hpp"

#include "mlir/lowering/ArithmeticOpLowering.hpp"
#include "mlir/lowering/ComparisonOpLowering.hpp"
#include "mlir/lowering/ConstOpLowering.hpp"
#include "mlir/lowering/EmitOpLowering.hpp"
#include "mlir/lowering/LogicalOpLowering.hpp"
#include "mlir/lowering/SelectOpLowering.hpp"

using namespace mlir;
using namespace ::voila::mlir;
using namespace ::voila::mlir::lowering;
void VoilaToAffineLoweringPass::runOnFunction()
{
    auto function = getFunction();

    // We only lower the main function as we expect that all other functions have
    // been inlined.
    if (function.getName() != "main")
        return;

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering.
    target.addLegalDialect<AffineDialect, memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect,
                           tosa::TosaDialect, linalg::LinalgDialect>();

    // We also define the dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Toy operations that don't want
    // to lower, `toy.print`, as `legal`.
    target.addIllegalDialect<::mlir::voila::VoilaDialect>();
    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    // constant lowerings
    patterns.add<AndOpLowering,OrOpLowering,BoolConstOpLowering, IntConstOpLowering, FltConstOpLowering,SelectOpLowering>(&getContext());
    // arithmetic lowerings
    patterns.add<AddIOpLowering, SubIOpLowering, MulIOpLowering, DivFOpLowering>(&getContext());
    // comparison lowerings
    patterns.add<EqIOpLowering, NeqIOpLowering, LeIOpLowering, LeqIOpLowering, GeIOpLowering, GeqIOpLowering>(&getContext());

    patterns.add<EmitOpLowering>(&getContext(), function);

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
    {
        function.dump();
        signalPassFailure();
    }
    // match function return type after emit lowering
    // FIXME: this is only a workaround, there must be a more clean way to achieve bufferization for function return
    auto newType = FunctionType::get(&getContext(), function.getType().getInputs(),
                                     function.body().back().back().getOperandTypes());
    function.setType(newType);
}

namespace voila::mlir
{
    /// Create a pass for lowering operations in the `Affine` and `Std` dialects.
    std::unique_ptr<Pass> createLowerToAffinePass()
    {
        return std::make_unique<VoilaToAffineLoweringPass>();
    }
} // namespace voila::mlir