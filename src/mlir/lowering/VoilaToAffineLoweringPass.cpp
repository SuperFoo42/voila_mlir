#include "mlir/lowering/VoilaToAffineLoweringPass.hpp"

#include "mlir/lowering/ConstOpLowering.hpp"
#include "mlir/lowering/ArithmeticOpLowering.hpp"
#include "mlir/lowering/EmitOpLowering.hpp"

void voila::mlir::lowering::VoilaToAffineLoweringPass::runOnFunction()
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
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `MemRef` and `Standard` dialects.
    target.addLegalDialect<AffineDialect, memref::MemRefDialect, StandardOpsDialect>();

    // We also define the dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Toy operations that don't want
    // to lower, `toy.print`, as `legal`.
    target.addIllegalDialect<::mlir::voila::VoilaDialect>();
    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<IntConstOpLowering, FltConstOpLowering, BoolConstOpLowering, EmitOpLowering, AndOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects.
std::unique_ptr<::mlir::Pass> voila::mlir::createLowerToAffinePass()
{
    return std::make_unique<voila::mlir::lowering::VoilaToAffineLoweringPass>();
}
