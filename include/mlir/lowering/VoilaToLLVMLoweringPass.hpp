#pragma once
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM//MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"

#include "llvm/ADT/Sequence.h"

namespace voila::mlir
{
    namespace lowering
    {
        struct VoilaToLLVMLoweringPass
            : public ::mlir::PassWrapper<VoilaToLLVMLoweringPass, ::mlir::OperationPass<::mlir::ModuleOp>>
        {
            void getDependentDialects(::mlir::DialectRegistry &registry) const override
            {
                registry.insert<::mlir::LLVM::LLVMDialect, ::mlir::scf::SCFDialect>();
            }
            void runOnOperation() final;
        };
    } // namespace lowering

    /**
     * Create a pass for lowering operations the remaining `Voila` operations, as
     * well as `Affine` and `Std`, to the LLVM dialect for codegen.
     * @return
     */
    std::unique_ptr<::mlir::Pass> createLowerToLLVMPass();
} // namespace voila::mlir