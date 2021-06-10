#pragma once

#include <stdexcept>

namespace voila
{
    class LLVMGenerationError : virtual public std::runtime_error
    {
      public:
        //TODO
        LLVMGenerationError() : std::runtime_error("Failed to emit LLVM IR") {}
        ~LLVMGenerationError() noexcept override = default;
    };
} // namespace voila::ast
