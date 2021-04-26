#pragma once
#include "IExpression.hpp"
#include "ASTVisitor.hpp"
#include "Expression.hpp"

namespace voila::ast
{

    /**
     * @brief Meta node to wrap expressions into predicates
     *
     */
    class Predicate : public IExpression
    {

      public:
        explicit Predicate(Location loc, Expression expr);

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_predicate() const final;

        Predicate * as_predicate() final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

      public:
        Expression expr;
    };

} // namespace voila::ast
