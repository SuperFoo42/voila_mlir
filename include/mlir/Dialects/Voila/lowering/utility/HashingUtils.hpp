#pragma once

#include "mlir/Dialects/Voila/IR/VoilaOps.h"   // for GatherOpAdaptor
#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef

namespace mlir
{
    class ImplicitLocOpBuilder;
}

namespace voila::mlir::lowering::utils
{
    ::mlir::Value createValueCmp(::mlir::ImplicitLocOpBuilder &builder,
                               const ::mlir::SmallVector<::mlir::Value> &vals,
                               const ::mlir::SmallVector<::mlir::Value> &toCmp);

    std::pair<::mlir::SmallVector<::mlir::Value>, ::mlir::Value> allocHashTables(::mlir::ImplicitLocOpBuilder &rewriter, ::mlir::ValueRange values);

    ::mlir::Value createKeyComparisons(::mlir::ImplicitLocOpBuilder &builder,
                                     const ::mlir::SmallVector<::mlir::Value> &hts,
                                     const ::mlir::SmallVector<::mlir::Value> &hashInvalidConsts,
                                     const ::mlir::SmallVector<::mlir::Value> &toStores,
                                     const ::mlir::ValueRange &idx);

    ::mlir::Value getHTSize(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value values);
} // namespace voila::mlir::lowering::utils