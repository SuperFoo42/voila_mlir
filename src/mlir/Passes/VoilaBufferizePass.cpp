#include "mlir/Passes/VoilaBufferizePass.hpp"

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"


namespace voila::mlir {
    using namespace ::mlir;
    using namespace bufferization;
    using namespace memref;

    namespace lowering {


        struct VoilaBufferizePass : public PassWrapper<VoilaBufferizePass, ::mlir::OperationPass<ModuleOp>> {
            MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VoilaBufferizePass)

            OneShotBufferizationOptions options;

            explicit VoilaBufferizePass(OneShotBufferizationOptions options) : options(std::move(options)) {}

            void getDependentDialects(::mlir::DialectRegistry &registry) const override {
                registry.insert<MemRefDialect, BufferizationDialect>();
            }

            void runOnOperation() final;
        };


        void VoilaBufferizePass::runOnOperation() {

            RewritePatternSet patterns(&getContext());

            // We want to completely lower to LLVM, so we use a `FullConversion`. This
            // ensures that only legal operations will remain after the conversion.
            auto module = getOperation();

            if (failed(runOneShotModuleBufferize(module, options)))
                signalPassFailure();
        }
    }

    std::unique_ptr<::mlir::Pass> createVoilaBufferizePass() {
        bufferization::OneShotBufferizationOptions bufferizationOps;
        bufferizationOps.allowReturnAllocs = true;
        bufferizationOps.allowUnknownOps = true;
        bufferizationOps.bufferizeFunctionBoundaries = true;
        //bufferizationOps.bufferAlignment = 128; TODO: buffer alignment option?
        bufferizationOps.createDeallocs = false; //not yet possible, as return values are also deallocated
        bufferizationOps.allowDialectInFilter<tensor::TensorDialect, linalg::LinalgDialect, arith::ArithmeticDialect>();
        return std::make_unique<lowering::VoilaBufferizePass>(std::move(bufferizationOps));
    }
} // voila