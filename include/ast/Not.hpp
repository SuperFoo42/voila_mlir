#pragma once
#include <utility>

#include "Expression.hpp"
#include "Logical.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class Not : public Logical
    {
      public:
        explicit Not(const Location loc,Expression expr) :
            Logical(loc), param(std::move(expr))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_not() const final;

        Not *as_not() final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression param;
    };
} // namespace voila::ast