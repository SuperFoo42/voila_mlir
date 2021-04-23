#pragma once
#include "Const.hpp"

#include <utility>
namespace voila::ast
{
    class StrConst : public Const
    {
      public:
        explicit StrConst(std::string val) : Const(), val{std::move(val)} {}

        [[nodiscard]] bool is_string() const final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        const std::string val;
    };
} // namespace voila::ast