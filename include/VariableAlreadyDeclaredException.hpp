#pragma once

#include <stdexcept>

namespace voila::ast
{
    class VariableAlreadyDeclaredException : virtual public std::runtime_error
    {
      public:
        VariableAlreadyDeclaredException() : std::runtime_error("Variable is already declared in scope.") {}
        ~VariableAlreadyDeclaredException() noexcept override = default;
    };
} // namespace voila::ast
