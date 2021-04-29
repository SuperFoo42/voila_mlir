#pragma once
#include "IExpression.hpp"
#include "ASTVisitor.hpp"
#include <utility>

namespace voila::ast
{
    class Variable : public IExpression
    {
      public:
        explicit Variable(const Location loc, std::string val) : IExpression(loc), var{std::move(val)} {}

        [[nodiscard]] bool is_variable() const final;

        Variable *as_variable() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::string var;
    };
} // namespace voila::ast