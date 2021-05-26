#pragma once
#include <stdexcept>
namespace voila
{
    class NotInferedException : public std::runtime_error
    {
      public:
        NotInferedException() : std::runtime_error("Type or shape of object not infered") {}
    };
} // namespace voila
