#pragma once
#include <utility>

#include "Expression.hpp"
#include "ASTVisitor.hpp"

namespace voila::ast
{
    class Gather : public IExpression
    {
      public:
        Gather(Expression lhs, Expression rhs) : column{std::move(lhs)}, idxs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] bool is_gather() const final;

        Gather *as_gather() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression column, idxs;
    };
} // namespace voila::ast