#pragma once

#include <stdexcept>

namespace voila
{
    class LLVMOptimizationError : virtual public std::runtime_error
    {
      public:
        //TODO
        LLVMOptimizationError(std::string &err) : std::runtime_error("Failed to optimize LLVM IR " + err) {}
        ~LLVMOptimizationError() noexcept override = default;
    };
} // namespace voila::ast
