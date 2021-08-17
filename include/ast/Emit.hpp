#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
    class Emit : public IStatement
    {
      public:
        explicit Emit(Location loc, std::vector<Expression> expr) : IStatement(loc), exprs{std::move(expr)} {}

        [[nodiscard]] bool is_emit() const final;

        Emit *as_emit() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::vector<Expression> exprs;
    };

} // namespace voila::ast