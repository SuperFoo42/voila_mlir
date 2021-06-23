#include "mlir/lowering/EmitOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::EmitOp;
    LogicalResult EmitOpLowering::matchAndRewrite(EmitOp op, PatternRewriter &rewriter) const
    {
        // lower to std::return
        // here we should only have to deal with the emit of the main function, since all other uses should have been
        // inlined
        SmallVector<Value> ops;
        for (auto o : op.getOperands())
        {
            ops.push_back(o);
        }

        rewriter.replaceOpWithNewOp<ReturnOp>(op, ops);
        return success();
    }
    EmitOpLowering::EmitOpLowering(MLIRContext *ctx, FuncOp &function) :
        OpRewritePattern<EmitOp>(ctx), function{function}
    {
    }
} // namespace voila::mlir::lowering