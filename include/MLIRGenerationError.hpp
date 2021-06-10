#pragma once
#include <stdexcept>
namespace voila
{
    class MLIRGenerationError : public std::runtime_error
    {
      public:
        MLIRGenerationError() : std::runtime_error("Error generating MLIR") {}
    };
} // namespace voila
