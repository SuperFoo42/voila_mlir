#pragma once

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Pass/Pass.h"
#include <memory>


namespace voila::mlir {
    std::unique_ptr<::mlir::Pass> createVoilaBufferizePass();
} // mlir
