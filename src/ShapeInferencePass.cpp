#include "mlir/ShapeInferencePass.hpp"
namespace voila::mlir
{
    using namespace ::mlir;

    static bool allOperandsInferred(Operation *op)
    {
        return llvm::all_of(op->getOperandTypes(),
                            [](::mlir::Type operandType) { return operandType.isa<RankedTensorType>(); });
    }

    /// A utility method that returns if the given operation has a dynamically
    /// shaped result.
    static bool returnsDynamicShape(Operation *op)
    {
        return llvm::any_of(op->getResultTypes(),
                            [](::mlir::Type resultType) { return !resultType.isa<RankedTensorType>(); });
    }

    void ShapeInferencePass::runOnFunction()
    {
        auto f = getFunction();

        // Populate the worklist with the operations that need shape inference:
        // these are operations that return a dynamic shape.
        ::mlir::Operation *emitOp = nullptr;
        llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
        f.walk(
            [&](mlir::Operation *op)
            {
                if (returnsDynamicShape(op))
                    opWorklist.insert(op);
                if (dyn_cast<::mlir::voila::EmitOp>(op))
                    emitOp = op;
            });

        // Iterate on the operations in the worklist until all operations have been
        // inferred or no change happened (fix point).
        while (!opWorklist.empty())
        {
            // Find the next operation ready for inference, that is an operation
            // with all operands already resolved (non-generic).
            auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
            if (nextop == opWorklist.end())
                break;

            Operation *op = *nextop;
            opWorklist.erase(op);

            // Ask the operation to infer its output shapes.
            ::mlir::voila::ShapeInference shapeOp = dyn_cast<::mlir::voila::ShapeInference>(op);
            if (shapeOp)
            {
                shapeOp.inferShapes();
            }
            else
            {
                throw NotInferedException();
            }
        }

        // If the operation worklist isn't empty, this indicates a failure.
        if (!opWorklist.empty())
        {
            f.emitError("Shape inference failed, ") << opWorklist.size() << " operations couldn't be inferred\n";
            signalPassFailure();
        }

        //infer function return type from emit op
        if (!f.getType().getResults().empty())
        {
            assert(emitOp);
            assert(f.getType().getResults().size() == emitOp->getOperands().size());
            f.setType(
                FunctionType::get(&this->getContext(), f.getType().getInputs(), emitOp->getOperands().getTypes()));
        }
        else
        {
            f.setType(
                FunctionType::get(&this->getContext(), f.getType().getInputs(), nullptr));
        }
    }

    std::unique_ptr<::mlir::Pass> createShapeInferencePass()
    {
        return std::make_unique<ShapeInferencePass>();
    }
} // namespace voila::mlir
