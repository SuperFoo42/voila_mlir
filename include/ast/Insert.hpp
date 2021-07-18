#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IExpression.hpp"

#include <string>

namespace voila::ast
{
    class Insert : public IExpression
    {
      public:
        Insert(Location loc, Expression keys) :
            IExpression(loc), keys{std::move(keys)}
        {
            // TODO
        }

        [[nodiscard]] bool is_insert() const final;

        Insert *as_insert() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression keys;
    };
} // namespace voila::ast