#pragma once
#include "Arithmetic.hpp"
#include "Expression.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class Mod : public Arithmetic
    {
      public:
        Mod(Expression lhs, Expression rhs) : Arithmetic(std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_mod() const final;

        Mod *as_mod() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast