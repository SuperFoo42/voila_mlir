#pragma once

#include <stdexcept>

namespace voila::ast
{
    class MLIRGenerationException : virtual public std::runtime_error
    {
      public:
        //TODO
        MLIRGenerationException() : std::runtime_error("") {}
        explicit MLIRGenerationException(std::string_view str) : std::runtime_error(str.data()) {}
        ~MLIRGenerationException() noexcept override = default;
    };
} // namespace voila::ast
