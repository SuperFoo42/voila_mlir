#pragma once

#include <stdexcept>

namespace voila::ast
{
    class PredicationUnsupportedException : virtual public std::runtime_error
    {
      public:
        explicit PredicationUnsupportedException(const std::string &msg) : std::runtime_error(msg) {}
        ~PredicationUnsupportedException() noexcept override = default;
    };
} // namespace voila::ast
