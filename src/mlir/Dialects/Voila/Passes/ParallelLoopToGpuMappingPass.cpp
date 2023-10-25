#include "mlir/Dialects/Voila/Passes/ParallelLoopToGpuMappingPass.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir
{
    class Operation;
}

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::func;
    namespace lowering
    {
        /*        StringRef ParallelLoopToGPUMappingPass::getArgument() const
                {
                    return "test-gpu-greedy-parallel-loop-mapping";
                }
                StringRef ParallelLoopToGPUMappingPass::getDescription() const
                {
                    return "Greedily maps all parallel loops to gpu hardware ids.";
                }*/
        void ParallelLoopToGPUMappingPass::runOnOperation()
        {

            // TODO:
            //Operation *op = getOperation();
            // for (auto &region : op->getRegions())
            //     greedilyMapParallelSCFToGPU(region);
        }
    } // namespace lowering
    std::unique_ptr<::mlir::Pass> createParallelLoopToGPUMappingPass()
    {
        return std::make_unique<lowering::ParallelLoopToGPUMappingPass>();
    }
} // namespace voila::mlir