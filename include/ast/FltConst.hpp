#pragma once
#include "Const.hpp"
namespace voila::ast
{
    class FltConst : public Const
    {
      public:
        explicit FltConst(const double val) : Const(), val{val} {}

        [[nodiscard]] bool is_float() const final;

        FltConst *as_float() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        const double val;
    };
} // namespace voila::ast