#pragma once
#include <memory>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "mlir/ShapeInferenceInterface.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"
#pragma GCC diagnostic pop
#include <NotInferedException.hpp>

namespace voila::mlir
{
    using namespace ::mlir;

    class ShapeInferencePass : public PassWrapper<ShapeInferencePass, FunctionPass>
    {


      public:
        void runOnFunction() override;
    };
} // namespace voila::mlir::shape_inference

namespace voila::mlir
{
    std::unique_ptr<::mlir::Pass> createShapeInferencePass();
} // namespace voila::mlir