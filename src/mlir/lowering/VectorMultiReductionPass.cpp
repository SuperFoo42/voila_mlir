#include "mlir/lowering/VectorMultiReductionPass.hpp"
namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::vector;
    std::unique_ptr<Pass> createLowerVectorMultiReductionPass()
    {
        return std::make_unique<VectorMultiReductionPass>();
    }
    StringRef VectorMultiReductionPass::getDescription() const
    {
        return "Test conversion patterns to lower vector.multi_reduction to other "
               "vector ops";
    }
    void VectorMultiReductionPass::runOnFunction()
    {
        RewritePatternSet patterns(&getContext());
        populateVectorMultiReductionLoweringPatterns(patterns, !useOuterReductions);
        (void) applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
    }
    void VectorMultiReductionPass::getDependentDialects(DialectRegistry &registry) const
    {
        registry.insert<memref::MemRefDialect>();
    }
} // namespace voila::mlir