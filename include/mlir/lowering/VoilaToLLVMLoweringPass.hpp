#pragma once
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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

using namespace mlir;

namespace voila::mlir::lowering
{
    struct VoilaToLLVMLoweringPass : public PassWrapper<VoilaToLLVMLoweringPass, OperationPass<ModuleOp>>
    {
        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
        }
        void runOnOperation() final;
    };
} // namespace voila::mlir::lowering

namespace voila::mlir
{
    /// Create a pass for lowering operations the remaining `Voila` operations, as
    /// well as `Affine` and `Std`, to the LLVM dialect for codegen.
    std::unique_ptr<::mlir::Pass> createLowerToLLVMPass();
} // namespace voila::mlir