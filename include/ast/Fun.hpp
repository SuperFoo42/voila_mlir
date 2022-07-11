#pragma once
#include "ASTNode.hpp"
#include "Statement.hpp"

#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <unordered_map>

namespace voila::ast
{
    class Fun : public ASTNode
    {
      public:

        Fun(Location loc, std::string fun, std::vector<Expression> args, std::vector<Statement> exprs);
        Fun() = default;
        Fun(Fun &) = default;
        Fun(const Fun &) = default;
        Fun(Fun &&) = default;

        Fun &operator=(const Fun &) = default;

        ~Fun() override = default;

        [[nodiscard]] bool is_function_definition() const override;

        Fun *as_function_definition() override;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &o) const override;
        void visit(ASTVisitor &visitor) const override;
        void visit(ASTVisitor &visitor) override;

        std::string name;
        std::vector<Expression> args;
        std::vector<Statement> body;
        std::optional<Statement> result;
        std::unordered_map<std::string, Expression> variables;
    };
} // namespace voila::ast