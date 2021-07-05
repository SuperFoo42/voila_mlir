#pragma once

#include <stdexcept>

namespace voila::ast
{
    class MLIRModuleVerificationError : virtual public std::runtime_error
    {
      public:
        //TODO
        MLIRModuleVerificationError() : std::runtime_error("module verification error") {}
        ~MLIRModuleVerificationError() noexcept override = default;
    };
} // namespace voila::ast
