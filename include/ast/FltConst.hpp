#pragma once
#include "Const.hpp"
namespace voila::ast
{
    class FltConst : Const
    {
      public:
        FltConst(const double val) : Const(), val{val} {}

        bool is_float() const final
        {
            return true;
        }

        FltConst *as_float() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "float";
        }

        const double val;
    };
} // namespace voila::ast