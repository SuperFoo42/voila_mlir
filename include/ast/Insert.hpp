#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IExpression.hpp"

#include <string>
#include <utility>

namespace voila::ast
{
    class Insert : public IExpression
    {
      public:
        Insert(Location loc, Expression keys, std::vector<Expression> values) :
            IExpression(loc), keys{std::move(keys)}, values{std::move(values)}
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
        std::vector<Expression> values;
    };
} // namespace voila::ast