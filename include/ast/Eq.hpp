#pragma once
#include "ASTVisitor.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Eq : public Comparison
    {
      public:
        Eq(Expression lhs, Expression rhs) : Comparison(std::move(lhs), std::move(rhs))
        {
            // TODO
        }
        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_eq() const final;

        Eq *as_eq() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast