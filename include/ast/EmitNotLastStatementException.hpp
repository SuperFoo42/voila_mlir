#pragma once

#include <stdexcept>

namespace voila::ast
{
    class EmitNotLastStatementException : virtual public std::runtime_error
    {
      public:
        EmitNotLastStatementException() : std::runtime_error("Emit is not last statement in function, or there exist multiple emit statements in function.") {}
        ~EmitNotLastStatementException() noexcept override = default;
    };
} // namespace voila::ast
