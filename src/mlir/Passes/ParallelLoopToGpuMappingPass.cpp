#include "mlir/Passes/ParallelLoopToGpuMappingPass.hpp"

#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
namespace voila::mlir
{
    using namespace ::mlir;
    namespace lowering
    {
        StringRef ParallelLoopToGPUMappingPass::getArgument() const
        {
            return "test-gpu-greedy-parallel-loop-mapping";
        }
        StringRef ParallelLoopToGPUMappingPass::getDescription() const
        {
            return "Greedily maps all parallel loops to gpu hardware ids.";
        }
        void ParallelLoopToGPUMappingPass::runOnOperation()
        {
            Operation *op = getOperation();
            for (auto &region : op->getRegions())
                greedilyMapParallelSCFToGPU(region);
        }
    } // namespace lowering
    std::unique_ptr<::mlir::Pass> createParallelLoopToGPUMappingPass()
    {
        return std::make_unique<lowering::ParallelLoopToGPUMappingPass>();
    }
} // namespace voila::mlir