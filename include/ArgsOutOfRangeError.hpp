#pragma once

#include <stdexcept>


namespace voila
{
    class ArgsOutOfRangeError : public std::out_of_range
    {
      public:
        explicit ArgsOutOfRangeError() : std::out_of_range("Already all parameters bound to function.") {}
        ~ArgsOutOfRangeError() noexcept override = default;
    };
} // namespace voila
