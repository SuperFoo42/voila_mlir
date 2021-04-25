#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <optional>
#include <utility>
#include <vector>
#include "ASTVisitor.hpp"

namespace voila::ast
{
    class Assign : public IStatement
    {
      public:
        Assign(std::string dest, Expression expr);

        [[nodiscard]] bool is_assignment() const final;

        Assign *as_assignment() final;

        [[nodiscard]] std::string type2string() const final
        {
            return "assignment";
        }

        void predicate(Expression expression) final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor);
        void visit(ASTVisitor &visitor) const;
      public:
        std::string dest;
        Expression expr;
        std::optional<Expression> pred;
    };

} // namespace voila::ast