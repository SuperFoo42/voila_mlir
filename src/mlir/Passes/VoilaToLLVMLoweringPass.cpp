#include "mlir/Passes/VoilaToLLVMLoweringPass.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM//MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
//===----------------------------------------------------------------------===//
// VoilaToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::func;
    namespace lowering
    {
        void VoilaToLLVMLoweringPass::runOnOperation()
        {
            // The first thing to define is the conversion target. This will define the
            // final target for this lowering. For this lowering, we are only targeting
            // the LLVM dialect.
            LLVMConversionTarget target(getContext());
            target.addLegalOp<ModuleOp>();

            // During this lowering, we will also be lowering the MemRef types, that are
            // currently being operated on, to a representation in LLVM. To perform this
            // conversion we use a TypeConverter as part of the lowering. This converter
            // details how one type maps to another. This is necessary now that we will be
            // doing more complicated lowerings, involving loop region arguments.
            LLVMTypeConverter typeConverter(&getContext());

            // Now that the conversion target has been defined, we need to provide the
            // patterns used for lowering. At this point of the compilation process, we
            // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
            // are already exists a set of patterns to transform `affine` and `std`
            // dialects. These patterns lowering in multiple stages, relying on transitive
            // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
            // patterns must be applied to fully transform an illegal operation into a
            // set of legal ones.
            RewritePatternSet patterns(&getContext());
            populateAffineToVectorConversionPatterns(patterns);
            populateAffineToStdConversionPatterns(patterns);
            populateSCFToControlFlowConversionPatterns(patterns);
            populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
            populateVectorToLLVMConversionPatterns(typeConverter, patterns, true);
            populateFuncToLLVMFuncOpConversionPattern(typeConverter, patterns);
            populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
            populateReconcileUnrealizedCastsPatterns(patterns);
            // We want to completely lower to LLVM, so we use a `FullConversion`. This
            // ensures that only legal operations will remain after the conversion.
            auto module = getOperation();
            if (failed(applyFullConversion(module, target, std::move(patterns))))
                signalPassFailure();
        }
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createLowerToLLVMPass()
    {
        return std::make_unique<lowering::VoilaToLLVMLoweringPass>();
    }
} // namespace voila::mlir
