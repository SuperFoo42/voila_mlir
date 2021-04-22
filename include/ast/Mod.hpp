#pragma once
#include "Arithmetic.hpp"
#include "BinaryOP.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Mod : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "mod";
        }

        bool is_mod() const final
        {
            return true;
        }

        Mod *as_mod() final
        {
            return this;
        }
    };
} // namespace voila::ast