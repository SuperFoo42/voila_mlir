#pragma once
#include <stdexcept>
namespace voila
{
    class NonMatchingArityException : public std::runtime_error
    {
      public:
        NonMatchingArityException() : std::runtime_error("Arity of types does not match") {}
    };
} // namespace voila
