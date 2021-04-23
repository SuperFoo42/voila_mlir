#pragma once
#include "ASTNode.hpp"
#include "Statement.hpp"

#include <vector>

namespace voila::ast
{
    class Fun : public ASTNode
    {
      public:
        Fun(std::string fun, std::vector<std::string> args, std::vector<Statement> exprs);
        Fun() = default;
        Fun(Fun &) = default;
        Fun(const Fun &) = default;
        Fun(Fun &&) = default;

        Fun &operator=(const Fun &) = default;

        [[nodiscard]] bool is_function_definition() const final;

        Fun *as_function_definition() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &o) const final;

        std::string name;
        std::vector<std::string> args;
        std::vector<Statement> body;
    };
} // namespace voila::ast