#include "mlir/lowering/LinalgTiledLoopsToAffineForPass.hpp"

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::linalg;
    namespace lowering
    {
        void LinalgTiledLoopsToAffineForPass::getDependentDialects(DialectRegistry &registry) const
        {
            registry.insert<LinalgDialect>();
        }
        void LinalgTiledLoopsToAffineForPass::runOnFunction()
        {
            RewritePatternSet patterns(&getContext());
            populateLinalgTiledLoopsToAffineForPatterns(patterns);
            (void) applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
        }

        ::mlir::StringRef LinalgTiledLoopsToAffineForPass::getDescription() const
        {
            return "Pass to lower linalg tiled loops into affine loops";
        }
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createConvertLinalgTiledLoopsToAffineForPass()
    {
        return std::make_unique<lowering::LinalgTiledLoopsToAffineForPass>();
    }
} // namespace voila::mlir