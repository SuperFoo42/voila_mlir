#pragma once

#include "ASTVisitor.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Leq : public Comparison
    {
      public:
        Leq(const Location loc, Expression lhs, Expression rhs) : Comparison(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_leq() const final;

        Leq *as_leq() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        using Comparison::clone;
    };
} // namespace voila::ast