#include "Expression.hpp"
#include "Arithmetic.hpp"
#include "ASTVisitor.hpp"
#pragma once

namespace voila::ast
{
    class Sub : public Arithmetic
    {
      public:
        Sub(const Location loc, Expression lhs, Expression rhs) : Arithmetic(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_sub() const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
        using Arithmetic::clone;
    };
}

