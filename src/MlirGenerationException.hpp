#pragma once

#include <stdexcept>

namespace voila::ast
{
    class MLIRGenerationException : virtual public std::runtime_error
    {
      public:
        //TODO
        MLIRGenerationException() : std::runtime_error("") {}
        ~MLIRGenerationException() noexcept override = default;
    };
} // namespace voila::ast
