#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Selection : public IExpression
    {
      public:
        Expression param;
        explicit Selection(const Location loc, Expression expr) : IExpression(loc), param(std::move(expr))
        {
            // TODO
        }

        [[nodiscard]] bool is_select() const final;

        Selection *as_select() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast