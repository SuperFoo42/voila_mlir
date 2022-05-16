#include "mlir/lowering/EmitOpLowering.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering {
    using namespace ::mlir;
    using namespace ::mlir::func;
    using ::mlir::voila::EmitOp;
    using ::mlir::voila::EmitOpAdaptor;

    LogicalResult EmitOpLowering::matchAndRewrite(EmitOp op, PatternRewriter &rewriter) const {
        EmitOpAdaptor adaptor(op);
        // lower to std::return
        // here we should only have to deal with the emit of the main function, since all other uses should have been
        // inlined

        rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.input());
        return success();
    }

    EmitOpLowering::EmitOpLowering(MLIRContext *ctx, FuncOp &function) :
            OpRewritePattern<EmitOp>(ctx), function{function} {
    }
} // namespace voila::mlir::lowering