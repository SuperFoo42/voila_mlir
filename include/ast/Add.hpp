#pragma once
#include "Arithmetic.hpp"
#include "Expression.hpp"
#include "ASTVisitor.hpp"

namespace voila::ast
{
    // Expressions
    class Add : public Arithmetic
    {
      public:
        Add(const Location loc, Expression lhs, Expression rhs) :
            Arithmetic(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_add() const final;

        Add *as_add() final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast