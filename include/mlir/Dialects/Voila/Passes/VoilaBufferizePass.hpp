#pragma once

#include <memory> // for unique_ptr
namespace mlir
{
    class Pass;
}

namespace voila::mlir
{
    std::unique_ptr<::mlir::Pass> createVoilaBufferizePass();
} // namespace voila::mlir
