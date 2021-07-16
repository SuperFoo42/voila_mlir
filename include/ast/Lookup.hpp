#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IExpression.hpp"

#include <string>

namespace voila::ast
{
    class Lookup : public IExpression
    {
      public:
        Lookup(Location loc, Expression table, Expression keys) :
            IExpression(loc), table{std::move(table)}, keys{std::move(keys)}
        {
            // TODO
        }

        [[nodiscard]] bool is_lookup() const final;

        Lookup *as_lookup() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression table, keys;
    };
} // namespace voila::ast