#pragma once
#include <utility>

#include "Expression.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class Read : public IExpression
    {
      public:
        Read(Location loc, Expression lhs, Expression rhs) :
            IExpression(loc), column{std::move(lhs)}, idx{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] bool is_read() const final;

        Read *as_read() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression column, idx;
    };
} // namespace voila::ast