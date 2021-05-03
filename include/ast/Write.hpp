
#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
    class Write : public IStatement
    {
      public:
        Write(Location loc, Expression src_col, Expression dest_col, Expression wpos);

        [[nodiscard]] bool is_write() const final;

        Write *as_write() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression dest;
        Expression start;
        Expression src;
    };

} // namespace voila::ast