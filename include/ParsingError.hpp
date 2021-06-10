#pragma once

#include <stdexcept>

namespace voila
{
    class ParsingError : virtual public std::runtime_error
    {
      public:
        explicit ParsingError() : std::runtime_error("Error parsing file.") {}
        ~ParsingError() noexcept override = default;
    };
}