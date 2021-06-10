#pragma once

#include <stdexcept>

namespace voila
{
    class MLIRLoweringError : virtual public std::runtime_error
    {
      public:
        //TODO
        MLIRLoweringError() : std::runtime_error("Error lowering mlir") {}
        ~MLIRLoweringError() noexcept override = default;
    };
} // namespace voila::ast
