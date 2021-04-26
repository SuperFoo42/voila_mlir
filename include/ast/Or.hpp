#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "Logical.hpp"

#include <utility>
namespace voila::ast
{
    class Or : public Logical
    {
      public:
        Or(Location loc, Expression lhs, Expression rhs) : Logical(loc), lhs{std::move(lhs)}, rhs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_or() const final;

        Or *as_or() final;

        Expression lhs, rhs;

        void visit(ASTVisitor &visitor);
        void visit(ASTVisitor &visitor) const;
    };
} // namespace voila::ast