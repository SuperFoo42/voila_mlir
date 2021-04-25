#pragma once
#include "Arithmetic.hpp"
#include "Expression.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class Div : public Arithmetic
    {
      public:
        Div(Expression lhs, Expression rhs) : Arithmetic(std::move(lhs),std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_div() const final;

        Div *as_div() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast