#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>
#include "ASTVisitor.hpp"

namespace voila::ast
{
    class Scatter : public IExpression
    {
      public:
        Scatter(Location loc, Expression idxs, Expression src_col);

        [[nodiscard]] bool is_scatter() const final;

        Scatter *as_scatter() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression idxs;
        Expression src;
    };

} // namespace voila::ast