#include "mlir/lowering/VoilaToLinalgLoweringPass.hpp"

namespace voila::mlir
{
    using namespace ::mlir;
    namespace lowering
    {
        void VoilaToLinalgLoweringPass::runOnFunction()
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
            target.addLegalDialect<AffineDialect, memref::MemRefDialect, StandardOpsDialect, linalg::LinalgDialect,
                                   tensor::TensorDialect, scf::SCFDialect>();

            // We also define the dialect as Illegal so that the conversion will fail
            // if any of these operations are *not* converted. Given that we actually want
            // a partial lowering, we explicitly mark the Toy operations that don't want
            // to lower, `toy.print`, as `legal`.
            target.addIllegalDialect<::mlir::voila::VoilaDialect>();
            target.addLegalOp<::mlir::voila::EmitOp, ::mlir::voila::BoolConstOp, ::mlir::voila::IntConstOp,
                              ::mlir::voila::FltConstOp, ::mlir::voila::SelectOp, ::mlir::voila::ReadOp,
                              ::mlir::voila::GatherOp, ::mlir::voila::MoveOp, ::mlir::voila::LoopOp>();
            // Now that the conversion target has been defined, we just need to provide
            // the set of patterns that will lower the Toy operations.
            RewritePatternSet patterns(&getContext());
            patterns.add<AndOpLowering, OrOpLowering, NotOpLowering, AddOpLowering, SubOpLowering, MulOpLowering,
                         DivOpLowering, ModOpLowering, EqOpLowering, NeqOpLowering, LeOpLowering, LeqOpLowering,
                         GeOpLowering, GeqOpLowering, HashOpLowering, LookupOpLowering, SumOpLowering, CountOpLowering,
                         InsertOpLowering, MinOpLowering, MaxOpLowering, AvgOpLowering>(&getContext());

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