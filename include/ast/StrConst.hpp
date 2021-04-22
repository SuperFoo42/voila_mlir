#pragma once
#include "Const.hpp"
namespace voila::ast
{
    class StrConst : Const
    {
      public:
        StrConst(const std::string &val) : Const(), val{val} {}

        bool is_string() const final
        {
            return true;
        }

        std::string type2string() const override
        {
            return "string";
        }

        const std::string val;
    };
} // namespace voila::ast