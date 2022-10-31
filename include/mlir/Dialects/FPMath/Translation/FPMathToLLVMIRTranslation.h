#pragma once

namespace mlir {

    class DialectRegistry;
    class MLIRContext;

/// Register the X86Vector dialect and the translation from it to the LLVM IR
/// in the given registry;
    void registerFPMathTranslation(DialectRegistry &registry);

/// Register the X86Vector dialect and the translation from it in the registry
/// associated with the given context.
    void registerFPMathTranslation(MLIRContext &context);

} // namespace mlir
