#pragma once

#include <memory>

namespace mlir
{
    class Pass;
}

namespace voila::mlir
{
    /**
     * Create a pass for lowering operations the remaining `Voila` operations, as
     * well as `Affine` and `Std`, to the LLVM dialect for codegen.
     * @return
     */
    std::unique_ptr<::mlir::Pass> createLowerVoilaToLLVMPass();
} // namespace voila::mlir