#pragma once

#include <stdexcept>

namespace voila
{
    class JITInvocationError : virtual public std::runtime_error
    {
      public:
        //TODO
        JITInvocationError() : std::runtime_error("JIT invocation failed") {}
        ~JITInvocationError() noexcept override = default;
    };
} // namespace voila::ast
