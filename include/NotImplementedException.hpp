#pragma once
#include <stdexcept>
namespace voila
{
    class NotImplementedException : public std::runtime_error
    {
      public:
        NotImplementedException() : std::runtime_error("Functionality not implemented") {}
        explicit NotImplementedException(const std::string &msg) : std::runtime_error("Functionality not implemented: " + msg) {}
    };
} // namespace voila
