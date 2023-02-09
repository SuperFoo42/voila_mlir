#include "mlir/Dialects/Voila/lowering/EmitOpLowering.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"    // for ReturnOp
#include "mlir/Dialects/Voila/IR/VoilaOps.h" // for EmitOp, EmitOpAdaptor
#include "mlir/Support/LLVM.h"               // for mlir
#include "llvm/ADT/StringRef.h"              // for operator==
#include "llvm/ADT/Twine.h"                  // for operator+

namespace mlir
{
    class MLIRContext;
}

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::func;
    using ::mlir::voila::EmitOp;
    using ::mlir::voila::EmitOpAdaptor;

    LogicalResult EmitOpLowering::matchAndRewrite(EmitOp op, PatternRewriter &rewriter) const
    {
        EmitOpAdaptor adaptor(op);
        // lower to std::return
        // here we should only have to deal with the emit of the main function, since all other uses should have been
        // inlined

        rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getInput());
        return success();
    }

    EmitOpLowering::EmitOpLowering(MLIRContext *ctx, FuncOp &function)
        : OpRewritePattern<EmitOp>(ctx), function{function}
    {
    }
} // namespace voila::mlir::lowering