#pragma once

#include <stdexcept>
namespace voila
{
    class IncompatibleTypesException : std::runtime_error
    {
      public:
        IncompatibleTypesException() : std::runtime_error("Types of objects not compatible") {}
    };
} // namespace voila
