#pragma once
#include "Const.hpp"
namespace voila::ast
{
    class IntConst : Const
    {
      public:
        IntConst(const std::intmax_t val) : Const(), val{val} {}

        bool is_integer() const final
        {
            return true;
        }

        IntConst *as_integer() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "integer";
        }

        const std::intmax_t val;
    };
} // namespace voila::ast