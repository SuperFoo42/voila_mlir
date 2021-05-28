#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IStatement.hpp"
#include "Predicate.hpp"
#include "Statement.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace voila::ast
{
    class Assign : public IStatement
    {
      public:
        Assign(Location loc, Expression dest, Statement expr);

        [[nodiscard]] bool is_assignment() const final;

        Assign *as_assignment() final;

        [[nodiscard]] std::string type2string() const final;

        void set_predicate(Expression expression) final;
        std::optional<Expression> get_predicate() final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) final;
        void visit(ASTVisitor &visitor) const final;
      public:
        Expression dest;
        Statement expr;
        std::optional<Expression> pred;
    };

} // namespace voila::ast