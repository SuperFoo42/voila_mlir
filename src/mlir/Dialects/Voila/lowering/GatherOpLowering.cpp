#include "mlir/Dialects/Voila/lowering/GatherOpLowering.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"             // for GenericOp
#include "mlir/Dialect/Tensor/IR/Tensor.h"             // for EmptyOp, Extr...
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"           // for GatherOpAdaptor
#include "mlir/IR/AffineMap.h"                         // for AffineMap
#include "mlir/IR/Builders.h"                          // for OpBuilder
#include "mlir/IR/BuiltinTypes.h"                      // for TensorType
#include "mlir/IR/Location.h"                          // for Location
#include "mlir/IR/OpDefinition.h"                      // for OpState
#include "mlir/IR/Operation.h"                         // for Operation
#include "mlir/IR/PatternMatch.h"                      // for PatternBenefit
#include "mlir/IR/TypeUtilities.h"                     // for getElementTyp...
#include "mlir/IR/Types.h"                             // for Type
#include "mlir/IR/ValueRange.h"                        // for ValueRange
#include "llvm/ADT/STLExtras.h"                        // for ValueOfRange
#include "llvm/ADT/SmallVector.h"                      // for SmallVector
#include "llvm/ADT/StringRef.h"                        // for operator==
#include "llvm/ADT/Twine.h"                            // for operator+

namespace mlir
{
    class MLIRContext;
    class Value;
} // namespace mlir

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::GatherOp;
    using ::mlir::voila::GatherOpAdaptor;

    GatherOpLowering::GatherOpLowering(MLIRContext *ctx) : ConversionPattern(GatherOp::getOperationName(), 1, ctx) {}

    LogicalResult GatherOpLowering::matchAndRewrite(Operation *op,
                                                    ArrayRef<Value> operands,
                                                    ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();

        GatherOpAdaptor gatherOpAdaptor(operands);

        auto out =
            rewriter
                .create<tensor::EmptyOp>(loc, gatherOpAdaptor.getIndices().getType().dyn_cast<TensorType>().getShape(),
                                         getElementTypeOrSelf(gatherOpAdaptor.getColumn()))
                ->getResults();

        auto gatherFunc = [&gatherOpAdaptor](OpBuilder &builder, Location loc, ValueRange vals)
        {
            auto idx = vals.front();
            auto res = builder.create<tensor::ExtractOp>(loc, gatherOpAdaptor.getColumn(), idx).getResult();
            builder.create<linalg::YieldOp>(loc, res);
        };

        llvm::SmallVector<AffineMap, 2> iter_maps(2, rewriter.getDimIdentityMap());

        auto linalgOp =
            rewriter.create<linalg::GenericOp>(loc, /*results*/ out.getType(),
                                               /*inputs*/ gatherOpAdaptor.getIndices(), /*outputs*/ out,
                                               /*indexing maps*/ iter_maps,
                                               /*iterator types*/ utils::IteratorType::parallel, gatherFunc);

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }
} // namespace voila::mlir::lowering