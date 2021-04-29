#pragma once
#include <stdexcept>
namespace voila
{
    class TypeNotInferedException : public std::runtime_error
    {
      public:
        TypeNotInferedException() : std::runtime_error("Type of object not infered") {}
    };
} // namespace voila
