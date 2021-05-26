#pragma once

namespace voila
{
    class MainFunctionNotFoundException : virtual public std::runtime_error
    {
      public:
        // TODO
        MainFunctionNotFoundException() : std::runtime_error("") {}
        ~MainFunctionNotFoundException() noexcept override = default;
    };
} // namespace voila
