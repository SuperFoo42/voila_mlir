#pragma once
#include "Const.hpp"
namespace voila::ast
{
    class BooleanConst : Const
    {
      public:
        BooleanConst(const bool val) : Const(), val{val} {}

        bool is_bool() const final
        {
            return true;
        }

        BooleanConst *as_bool() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "bool";
        }

        const bool val;
    };
} // namespace voila::ast