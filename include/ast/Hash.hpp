#pragma once

#include "ASTVisitor.hpp"
#include "Comparison.hpp"
#include "IExpression.hpp"

#include <utility>
namespace voila::ast
{
    class Hash : public IExpression
    {
      public:
        Expression items;

        Hash(const Location loc, Expression items) : IExpression(loc), items{std::move(items)} {}

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_hash() const final;

        Hash *as_hash() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
        void print(std::ostream &ostream) const override;
    };
} // namespace voila::ast