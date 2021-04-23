#pragma once
#include "Const.hpp"
namespace voila::ast
{
    class BooleanConst : public Const
    {
      public:
        explicit BooleanConst(const bool val) : Const(), val{val} {}

        [[nodiscard]] bool is_bool() const final;

        BooleanConst *as_bool() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        const bool val;
    };
} // namespace voila::ast