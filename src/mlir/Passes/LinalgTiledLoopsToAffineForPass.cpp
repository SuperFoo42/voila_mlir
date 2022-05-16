#include "mlir/Passes/LinalgTiledLoopsToAffineForPass.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace voila::mlir
{
    using namespace ::mlir;
    using namespace linalg;
    using namespace func;
    namespace lowering
    {
        void LinalgTiledLoopsToAffineForPass::getDependentDialects(DialectRegistry &registry) const
        {
            registry.insert<LinalgDialect>();
        }
        void LinalgTiledLoopsToAffineForPass::runOnOperation()
        {
            RewritePatternSet patterns(&getContext());
            //populateLinalgTiledLoopsToAffineForPatterns(patterns);
            (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
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