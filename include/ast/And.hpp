#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "Logical.hpp"

#include <utility>
namespace voila::ast
{
    class And : public Logical
    {
      public:
        And(const Location loc, Expression lhs, Expression rhs) :
            Logical(loc), lhs{std::move(lhs)}, rhs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_and() const final;

        And *as_and() final;

        Expression lhs, rhs;
        void visit(ASTVisitor &visitor);
        void visit(ASTVisitor &visitor) const;
    };
} // namespace voila::ast